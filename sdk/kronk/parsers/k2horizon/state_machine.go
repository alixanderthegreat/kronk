package k2horizon

import (
	"slices"
	"strings"

	"uuid"

	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

const (
	toolCallsOpen  = "<ifm|tool_calls>"
	toolCallsClose = "</ifm|tool_calls>"
	toolCallOpen   = "<ifm|tool_call>"
	toolCallClose  = "</ifm|tool_call>"
	argKeyOpen     = "<ifm|arg_key>"
)

// reasoningOpeners and reasoningClosers are the effort-specific reasoning
// tags.
var (
	reasoningOpeners = []string{"<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>"}
	reasoningClosers = []string{"</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"}
)

// endOfTurn are the markers that end generation. <|ifm|im_end|> closes an
// assistant turn and is one of the model's two EOS ids; a GGUF that lists
// only <|ifm|endoftext|> as end-of-generation would otherwise let it leak
// into content as text.
var endOfTurn = []string{"<|ifm|im_end|>", "<|ifm|endoftext|>"}

// Markers the scanner looks for, by state. Inside a tool-call block only
// the block closer and end of turn matter; every other tag there belongs to
// the call body and is left for parseK2.
var (
	outsideToolMarkers = slices.Concat(reasoningOpeners, reasoningClosers, []string{toolCallsOpen, toolCallOpen}, endOfTurn)
	insideToolMarkers  = slices.Concat([]string{toolCallsClose}, endOfTurn)
)

// stateMachine is a per-slot streaming state machine for K2-Horizon.
//
// Every tag is a single token in K2-Horizon's vocabulary, but decoded pieces
// are not guaranteed to line up with tags, so the state machine scans text
// rather than matching whole pieces: text that could still become a tag is
// carried into the next piece, and a piece that crosses channels queues one
// result per channel. Classify returns the oldest queued result; Flush
// drains the rest at end of generation.
//
// Inside a tool-call block everything but the block's closing tag goes to
// the tool channel, so parseK2 sees the whole block.
//
// Reasoning text is held back until the state machine knows what it is.
// The chat template opens a reasoning tag in every prompt, but the model
// does not always use it: after a tool result it often answers directly and
// ends the turn without ever closing the tag. Like IFM's own k2_horizon
// parser, text in a reasoning block that is never closed and never followed
// by a tool block is the answer. So held text is released as reasoning when
// a closer or a tool block arrives, and as the answer when the model ends
// its turn. If generation stops without the model ending its turn (e.g.
// max_tokens), it stays reasoning: a cut-off thought is not a reply.
type stateMachine struct {
	status model.Channel

	carry        string // Text that may be the start of a tag.
	held         strings.Builder
	sawEndOfTurn bool
	queue        []model.Result

	inToolBlock bool
	toolBuf     strings.Builder

	toolCallDeltas []model.ResponseToolCallDelta
	startedCalls   []model.ResponseToolCallDelta
}

// Reset returns the stateMachine to its initial state for reuse on a new
// request.
func (sm *stateMachine) Reset() {
	sm.status = model.ChannelAnswer
	sm.carry = ""
	sm.held.Reset()
	sm.sawEndOfTurn = false
	sm.queue = nil
	sm.inToolBlock = false
	sm.toolBuf.Reset()
	sm.toolCallDeltas = nil
	sm.startedCalls = nil
}

// Classify classifies a single decoded piece.
//
// Behavior is undefined if Classify is called after a previous call returned
// eog=true. Reset must be invoked between requests.
func (sm *stateMachine) Classify(content string) (model.Result, bool) {
	eog := sm.scan(content)

	if len(sm.queue) > 0 {
		result := sm.queue[0]
		sm.queue = sm.queue[1:]
		return result, eog
	}

	// Nothing to emit. While reasoning text is held, still report the
	// reasoning channel so the slot accounts tokens (and skips grammar)
	// as it would for streamed reasoning.
	if sm.status == model.ChannelReasoning && !eog {
		return model.Result{Channel: model.ChannelReasoning}, false
	}

	return model.Result{}, eog
}

// scan consumes text, queuing results, and reports whether the model ended
// its turn.
func (sm *stateMachine) scan(content string) bool {
	text := sm.carry + content
	sm.carry = ""

	for text != "" {
		markers := outsideToolMarkers
		if sm.inToolBlock {
			markers = insideToolMarkers
		}

		at, marker := firstMarker(text, markers)
		if at == -1 {
			keep := partialMarkerSuffix(text, markers)
			sm.emit(text[:len(text)-keep])
			sm.carry = text[len(text)-keep:]
			return false
		}

		sm.emit(text[:at])
		text = text[at+len(marker):]

		if slices.Contains(endOfTurn, marker) {
			sm.sawEndOfTurn = true
			return true
		}
		sm.handleMarker(marker)
	}

	return false
}

func (sm *stateMachine) handleMarker(marker string) {
	switch {
	case slices.Contains(reasoningOpeners, marker):
		sm.status = model.ChannelReasoning

	case slices.Contains(reasoningClosers, marker):
		sm.status = model.ChannelAnswer
		sm.releaseHeld(model.ChannelReasoning)

	case marker == toolCallsOpen, marker == toolCallOpen:
		// A tool block ends reasoning just as a closer does. A bare
		// <ifm|tool_call> is a call without the enclosing wrapper, and
		// stays in the buffer for parseK2.
		sm.releaseHeld(model.ChannelReasoning)
		sm.status = model.ChannelTool
		sm.inToolBlock = true
		sm.toolBuf.Reset()
		if marker == toolCallOpen {
			sm.emit(marker)
		}

	case marker == toolCallsClose:
		sm.inToolBlock = false
		sm.status = model.ChannelAnswer
	}
}

// emit routes text by the current state: tool-block text to the tool
// channel, reasoning to the held buffer, anything else to the answer.
func (sm *stateMachine) emit(text string) {
	if text == "" {
		return
	}

	switch {
	case sm.inToolBlock:
		sm.toolBuf.WriteString(text)
		sm.updateToolCallDeltas()
		sm.enqueue(model.ChannelTool, text)

	case sm.status == model.ChannelReasoning:
		sm.held.WriteString(text)

	default:
		sm.enqueue(sm.status, text)
	}
}

// enqueue appends a result, merging it into the previous one when both are
// on the same channel.
func (sm *stateMachine) enqueue(channel model.Channel, text string) {
	if n := len(sm.queue); n > 0 && sm.queue[n-1].Channel == channel {
		sm.queue[n-1].Content += text
		return
	}
	sm.queue = append(sm.queue, model.Result{Channel: channel, Content: text})
}

// releaseHeld queues held reasoning text on the given channel.
func (sm *stateMachine) releaseHeld(channel model.Channel) {
	if sm.held.Len() == 0 {
		return
	}
	sm.enqueue(channel, sm.held.String())
	sm.held.Reset()
}

// ConsumeVocabEOG records that the model ended its turn when the vocabulary
// itself classifies the end-of-turn token as end-of-generation, so Classify
// never sees it.
func (sm *stateMachine) ConsumeVocabEOG(string) {
	sm.sawEndOfTurn = true
}

// Flush drains what is left at end of generation, one result per call:
// queued results first, then any carried partial tag as ordinary text, then
// held reasoning (as the answer if the model ended its turn without closing
// the reasoning block, otherwise as reasoning).
func (sm *stateMachine) Flush() model.Result {
	if sm.carry != "" {
		carry := sm.carry
		sm.carry = ""
		sm.emit(carry)
	}

	if sm.held.Len() > 0 {
		if sm.sawEndOfTurn {
			sm.releaseHeld(model.ChannelAnswer)
		} else {
			sm.releaseHeld(model.ChannelReasoning)
		}
	}

	if len(sm.queue) == 0 {
		return model.Result{}
	}
	result := sm.queue[0]
	sm.queue = sm.queue[1:]
	return result
}

// ToolCallDeltas drains OpenAI-compatible tool-call activity deltas produced
// by the most recent Classify call.
func (sm *stateMachine) ToolCallDeltas() []model.ResponseToolCallDelta {
	deltas := sm.toolCallDeltas
	sm.toolCallDeltas = nil
	return deltas
}

// StartedToolCalls returns the tool-call identities emitted during the
// current request.
func (sm *stateMachine) StartedToolCalls() []model.ResponseToolCallDelta {
	return sm.startedCalls
}

// updateToolCallDeltas emits one activity delta per call once its function
// name is fully known.
func (sm *stateMachine) updateToolCallDeltas() {
	calls := strings.Split(sm.toolBuf.String(), toolCallOpen)[1:]
	for i := len(sm.startedCalls); i < len(calls); i++ {
		name, ok := streamingCallName(calls[i])
		if !ok {
			return
		}

		delta := model.ResponseToolCallDelta{
			ID:    newToolCallID(),
			Index: i,
			Type:  "function",
			Function: model.ResponseToolCallDeltaFunction{
				Name: name,
			},
		}
		sm.toolCallDeltas = append(sm.toolCallDeltas, delta)
		sm.startedCalls = append(sm.startedCalls, delta)
	}
}

// streamingCallName extracts a call's function name from a partially
// buffered call body, reporting false until the name is complete.
func streamingCallName(body string) (string, bool) {
	trimmed := strings.TrimLeft(body, " \t\r\n")
	if strings.HasPrefix(trimmed, "{") {
		return jsonCallName(trimmed)
	}

	// xml / xml_typed: the name runs up to the first newline or tag.
	end := len(trimmed)
	complete := false
	for _, stop := range []string{"\n", argKeyOpen, toolCallClose} {
		if at := strings.Index(trimmed, stop); at != -1 && at < end {
			end = at
			complete = true
		}
	}
	if !complete {
		return "", false
	}

	name := strings.TrimSpace(trimmed[:end])
	return name, name != ""
}

// firstMarker returns the earliest marker in text and its index, or -1.
// Where two markers start at the same index the longer one wins, so
// <ifm|think_fast> is not read as <ifm|think> plus text.
func firstMarker(text string, markers []string) (int, string) {
	best, found := -1, ""
	for _, marker := range markers {
		at := strings.Index(text, marker)
		if at == -1 {
			continue
		}
		if best == -1 || at < best || (at == best && len(marker) > len(found)) {
			best, found = at, marker
		}
	}
	return best, found
}

// partialMarkerSuffix returns the length of the longest suffix of text that
// is a proper prefix of one of the markers, i.e. text that may still become
// a marker once the next piece arrives.
func partialMarkerSuffix(text string, markers []string) int {
	longest := 0
	for _, marker := range markers {
		for n := min(len(marker)-1, len(text)); n > longest; n-- {
			if strings.HasSuffix(text, marker[:n]) {
				longest = n
				break
			}
		}
	}
	return longest
}

func newToolCallID() string {
	return "call_" + uuid.New().String()
}
