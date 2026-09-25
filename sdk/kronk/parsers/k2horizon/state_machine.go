package k2horizon

import (
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
// tags. Each is a single token in K2-Horizon's vocabulary.
var (
	reasoningOpeners = []string{"<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>"}
	reasoningClosers = []string{"</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"}
)

// endOfTurn are the markers that end generation. <|ifm|im_end|> closes an
// assistant turn and is one of the model's two EOS ids; a GGUF that lists
// only <|ifm|endoftext|> as end-of-generation would otherwise let it leak
// into content as text.
var endOfTurn = []string{"<|ifm|im_end|>", "<|ifm|endoftext|>"}

// stateMachine is a per-slot streaming state machine for K2-Horizon.
//
// Every marker it recognizes is a single vocabulary token, so matching is
// exact per decoded piece. Inside a tool-call block every piece, markers
// included, is routed to the tool channel so parseK2 sees the whole block.
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

	held         strings.Builder
	sawEndOfTurn bool

	inToolBlock bool
	toolBuf     strings.Builder
	// pendingTool is tool-block text not yet returned on the tool channel,
	// because the piece that produced it returned held reasoning instead.
	pendingTool string

	toolCallDeltas []model.ResponseToolCallDelta
	startedCalls   []model.ResponseToolCallDelta
}

// Reset returns the stateMachine to its initial state for reuse on a new
// request.
func (sm *stateMachine) Reset() {
	sm.status = model.ChannelAnswer
	sm.held.Reset()
	sm.sawEndOfTurn = false
	sm.inToolBlock = false
	sm.toolBuf.Reset()
	sm.pendingTool = ""
	sm.toolCallDeltas = nil
	sm.startedCalls = nil
}

// Classify classifies a single decoded token's content.
//
// Behavior is undefined if Classify is called after a previous call returned
// eog=true. Reset must be invoked between requests.
func (sm *stateMachine) Classify(content string) (model.Result, bool) {
	if isOneOf(content, endOfTurn) {
		sm.sawEndOfTurn = true
		return model.Result{}, true
	}

	if sm.inToolBlock {
		pending := sm.pendingTool
		sm.pendingTool = ""

		if content == toolCallsClose {
			sm.inToolBlock = false
			sm.status = model.ChannelAnswer
			return model.Result{Channel: model.ChannelTool, Content: pending}, false
		}

		sm.toolBuf.WriteString(content)
		sm.updateToolCallDeltas()
		return model.Result{Channel: model.ChannelTool, Content: pending + content}, false
	}

	switch {
	case isOneOf(content, reasoningOpeners):
		sm.status = model.ChannelReasoning
		return model.Result{}, false

	case isOneOf(content, reasoningClosers):
		sm.status = model.ChannelAnswer
		return sm.releaseHeld(model.ChannelReasoning), false

	case content == toolCallsOpen, content == toolCallOpen:
		// A tool block ends reasoning just as a closer does. A bare
		// <ifm|tool_call> is a call without the enclosing wrapper.
		held := sm.releaseHeld(model.ChannelReasoning)
		opener := ""
		if content == toolCallOpen {
			opener = content
		}
		sm.startToolBlock(opener)
		if held.Content != "" {
			// One result per piece: the held reasoning goes out now and the
			// opener rides along with the next tool-channel result.
			sm.pendingTool = opener
			return held, false
		}
		return model.Result{Channel: model.ChannelTool, Content: opener}, false
	}

	if sm.status == model.ChannelReasoning {
		sm.held.WriteString(content)
		return model.Result{}, false
	}

	return model.Result{Channel: sm.status, Content: content}, false
}

// releaseHeld drains held reasoning text on the given channel.
func (sm *stateMachine) releaseHeld(channel model.Channel) model.Result {
	if sm.held.Len() == 0 {
		return model.Result{}
	}
	content := sm.held.String()
	sm.held.Reset()
	return model.Result{Channel: channel, Content: content}
}

// ConsumeVocabEOG records that the model ended its turn when the vocabulary
// itself classifies the end-of-turn token as end-of-generation, so Classify
// never sees it.
func (sm *stateMachine) ConsumeVocabEOG(string) {
	sm.sawEndOfTurn = true
}

// Flush releases reasoning text still held at end of generation: as the
// answer if the model ended its turn without closing the reasoning block,
// otherwise (generation was cut short) as reasoning.
func (sm *stateMachine) Flush() model.Result {
	if sm.pendingTool != "" {
		pending := sm.pendingTool
		sm.pendingTool = ""
		return model.Result{Channel: model.ChannelTool, Content: pending}
	}
	if sm.sawEndOfTurn {
		return sm.releaseHeld(model.ChannelAnswer)
	}
	return sm.releaseHeld(model.ChannelReasoning)
}

func (sm *stateMachine) startToolBlock(content string) {
	sm.status = model.ChannelTool
	sm.inToolBlock = true
	sm.toolBuf.Reset()
	sm.toolBuf.WriteString(content)
	sm.updateToolCallDeltas()
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

	// xml / xml_typed: the name runs up to the first newline or marker.
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

func isOneOf(content string, markers []string) bool {
	for _, marker := range markers {
		if content == marker {
			return true
		}
	}
	return false
}

func newToolCallID() string {
	return "call_" + uuid.New().String()
}
