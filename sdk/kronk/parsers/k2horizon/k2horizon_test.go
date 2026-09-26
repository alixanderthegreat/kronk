package k2horizon

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

type step struct {
	token   string
	channel model.Channel
	content string
	eog     bool
}

func runSteps(t *testing.T, name string, c model.StateMachine, steps []step) {
	t.Helper()
	for i, s := range steps {
		got, eog := c.Classify(s.token)
		if got.Channel != s.channel {
			t.Errorf("%s step %d (%q): channel = %v, want %v",
				name, i, s.token, got.Channel, s.channel)
		}
		if got.Content != s.content {
			t.Errorf("%s step %d (%q): content = %q, want %q",
				name, i, s.token, got.Content, s.content)
		}
		if eog != s.eog {
			t.Errorf("%s step %d (%q): eog = %v, want %v",
				name, i, s.token, eog, s.eog)
		}
	}
}

// toolBuffer feeds tokens through a fresh state machine and returns what
// the batch engine would accumulate into its tool-call buffer.
func toolBuffer(t *testing.T, tokens []string) (string, *stateMachine) {
	t.Helper()
	sm := Parser{}.NewStateMachine().(*stateMachine)
	var buf strings.Builder
	for _, token := range tokens {
		got, eog := sm.Classify(token)
		if eog {
			break
		}
		if got.Channel == model.ChannelTool {
			buf.WriteString(got.Content)
		}
	}
	for got := sm.Flush(); got != (model.Result{}); got = sm.Flush() {
		if got.Channel == model.ChannelTool {
			buf.WriteString(got.Content)
		}
	}
	return buf.String(), sm
}

// =============================================================================
// Parser selection
// =============================================================================

func TestNew_ClaimsK2Horizon(t *testing.T) {
	tests := []struct {
		name string
		fp   model.Fingerprint
		want bool
	}{
		{"arch", model.Fingerprint{Architecture: "k2-horizon"}, true},
		{"arch-underscore", model.Fingerprint{Architecture: "k2_horizon"}, true},
		{"arch-mixed-case", model.Fingerprint{Architecture: "K2-Horizon"}, true},
		{"template-tool-call", model.Fingerprint{ChatTemplate: "<ifm|tool_calls>\n<ifm|tool_call>"}, true},
		{"template-think", model.Fingerprint{ChatTemplate: "<|ifm|im_start|>assistant\n<ifm|think>\n"}, true},
		{"name", model.Fingerprint{ModelName: "K2-Horizon-MoVA-36B-A4B"}, true},

		{"qwen", model.Fingerprint{Architecture: "qwen3", ChatTemplate: "<think>\n<tool_call>", ModelName: "Qwen3-8B"}, false},
		{"glm-markers", model.Fingerprint{ChatTemplate: "<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value>"}, false},
		{"empty", model.Fingerprint{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, got := New(tt.fp); got != tt.want {
				t.Errorf("New(%+v) claimed = %v, want %v", tt.fp, got, tt.want)
			}
		})
	}
}

// =============================================================================
// Reasoning and end of turn
// =============================================================================

func TestStateMachine_ReasoningPrimedByPrompt(t *testing.T) {
	for _, effort := range []struct{ open, close string }{
		{"<ifm|think>", "</ifm|think>"},
		{"<ifm|think_fast>", "</ifm|think_fast>"},
		{"<ifm|think_faster>", "</ifm|think_faster>"},
	} {
		sm := Parser{}.NewStateMachine()
		runSteps(t, effort.open, sm, []step{
			// The chat template leaves the opener at the end of the prompt;
			// the batch engine feeds it in before generation starts.
			// Reasoning is held until the closer decides what it was; the
			// channel is still reported so the slot accounts the tokens.
			{effort.open, model.ChannelReasoning, "", false},
			{"The user", model.ChannelReasoning, "", false},
			{" wants pong.", model.ChannelReasoning, "", false},
			{effort.close, model.ChannelReasoning, "The user wants pong.", false},
			{"\n", model.ChannelAnswer, "\n", false},
			{"pong", model.ChannelAnswer, "pong", false},
			{"<|ifm|im_end|>", model.ChannelNone, "", true},
		})
	}
}

func TestStateMachine_ThinkingDisabled(t *testing.T) {
	// enable_thinking=false renders an empty <ifm|think></ifm|think> in the
	// prompt, so generation starts directly in the answer.
	runSteps(t, "no-think", Parser{}.NewStateMachine(), []step{
		{"pong", model.ChannelAnswer, "pong", false},
		{"<|ifm|endoftext|>", model.ChannelNone, "", true},
	})
}

func TestStateMachine_Reset(t *testing.T) {
	sm := Parser{}.NewStateMachine()
	sm.Classify("<ifm|think>")
	sm.Classify("<ifm|tool_calls>")
	sm.Reset()
	runSteps(t, "after-reset", sm, []step{
		{"hello", model.ChannelAnswer, "hello", false},
	})
}

// =============================================================================
// Tool calls
// =============================================================================

func TestToolCall_JSON(t *testing.T) {
	buf, sm := toolBuffer(t, []string{
		"<ifm|think>", "Check the weather.", "</ifm|think>", "\n",
		"<ifm|tool_calls>", "\n",
		"<ifm|tool_call>", `{"name": "get_weather", `, `"arguments": {"city": "Paris", "days": 3}}`, "</ifm|tool_call>", "\n",
		"<ifm|tool_call>", `{"name": "get_time", "arguments": {}}`, "</ifm|tool_call>", "\n",
		"</ifm|tool_calls>",
		"<|ifm|im_end|>",
	})

	calls := Parser{}.ToolCall(context.Background(), nil, buf)
	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2: %+v", len(calls), calls)
	}
	if calls[0].Function.Name != "get_weather" || calls[1].Function.Name != "get_time" {
		t.Errorf("names = %q, %q", calls[0].Function.Name, calls[1].Function.Name)
	}
	if got := calls[0].Function.Arguments["city"]; got != "Paris" {
		t.Errorf("city = %#v, want Paris", got)
	}
	if got := calls[0].Function.Arguments["days"]; got != json.Number("3") {
		t.Errorf("days = %#v, want json.Number 3", got)
	}
	if len(calls[1].Function.Arguments) != 0 {
		t.Errorf("get_time arguments = %#v, want empty", calls[1].Function.Arguments)
	}

	started := sm.StartedToolCalls()
	if len(started) != 2 || started[0].Function.Name != "get_weather" || started[1].Index != 1 {
		t.Errorf("started deltas = %+v", started)
	}
}

func TestToolCall_JSONStringArguments(t *testing.T) {
	buf := `<ifm|tool_call>{"name": "f", "arguments": "{\"a\": 1}"}</ifm|tool_call>`
	calls := parseK2(buf)
	if len(calls) != 1 || calls[0].Status != 0 || calls[0].Function.Arguments["a"] != json.Number("1") {
		t.Errorf("calls = %+v", calls)
	}
}

func TestToolCall_XMLWithSchema(t *testing.T) {
	buf, sm := toolBuffer(t, []string{
		"<ifm|tool_calls>", "\n",
		"<ifm|tool_call>", "search", "\n",
		"<ifm|arg_key>", "query", "</ifm|arg_key>", "\n",
		"<ifm|arg_value>", "go 1.27 release", "</ifm|arg_value>", "\n",
		"<ifm|arg_key>", "limit", "</ifm|arg_key>", "\n",
		"<ifm|arg_value>", "5", "</ifm|arg_value>", "\n",
		"<ifm|arg_key>", "fresh", "</ifm|arg_key>", "\n",
		"<ifm|arg_value>", "true", "</ifm|arg_value>", "\n",
		"<ifm|arg_key>", "sites", "</ifm|arg_key>", "\n",
		"<ifm|arg_value>", `["go.dev"]`, "</ifm|arg_value>", "\n",
		"</ifm|tool_call>", "\n",
		"</ifm|tool_calls>",
	})

	tools := []model.D{{
		"type": "function",
		"function": model.D{
			"name": "search",
			"parameters": model.D{
				"type": "object",
				"properties": model.D{
					"query": model.D{"type": "string"},
					"limit": model.D{"type": "integer"},
					"fresh": model.D{"type": "boolean"},
					"sites": model.D{"type": "array"},
				},
			},
		},
	}}

	calls := Parser{}.ToolCallWithSchema(context.Background(), nil, buf, tools)
	if len(calls) != 1 || calls[0].Status != 0 {
		t.Fatalf("calls = %+v", calls)
	}
	args := calls[0].Function.Arguments
	if args["query"] != "go 1.27 release" {
		t.Errorf("query = %#v", args["query"])
	}
	if args["limit"] != json.Number("5") {
		t.Errorf("limit = %#v, want json.Number 5", args["limit"])
	}
	if args["fresh"] != true {
		t.Errorf("fresh = %#v, want true", args["fresh"])
	}
	if sites, ok := args["sites"].([]any); !ok || len(sites) != 1 || sites[0] != "go.dev" {
		t.Errorf("sites = %#v", args["sites"])
	}

	if started := sm.StartedToolCalls(); len(started) != 1 || started[0].Function.Name != "search" {
		t.Errorf("started deltas = %+v", started)
	}
}

func TestToolCall_XMLTyped(t *testing.T) {
	buf := "<ifm|tool_call>set\n" +
		"<ifm|arg_key>n</ifm|arg_key>\n<ifm|arg_type>integer</ifm|arg_type>\n<ifm|arg_value>42</ifm|arg_value>\n" +
		"<ifm|arg_key>label</ifm|arg_key>\n<ifm|arg_type>string</ifm|arg_type>\n<ifm|arg_value>007</ifm|arg_value>\n" +
		"</ifm|tool_call>"

	calls := parseK2(buf)
	if len(calls) != 1 || calls[0].Status != 0 {
		t.Fatalf("calls = %+v", calls)
	}
	if got := calls[0].Function.Arguments["n"]; got != json.Number("42") {
		t.Errorf("n = %#v, want json.Number 42", got)
	}
	if got := calls[0].Function.Arguments["label"]; got != "007" {
		t.Errorf("label = %#v, want the string 007", got)
	}
}

func TestToolCall_BareCallWithoutWrapper(t *testing.T) {
	buf, _ := toolBuffer(t, []string{
		"<ifm|tool_call>", `{"name": "ping", "arguments": {}}`, "</ifm|tool_call>",
		"<|ifm|im_end|>",
	})
	calls := parseK2(buf)
	if len(calls) != 1 || calls[0].Function.Name != "ping" {
		t.Errorf("calls = %+v", calls)
	}
}

func TestToolCall_Unterminated(t *testing.T) {
	calls := parseK2("\n<ifm|tool_call>ping\n<ifm|arg_key>host</ifm|arg_key>\n<ifm|arg_value>core</ifm|arg_value>\n")
	if len(calls) != 1 || calls[0].Status != 0 || calls[0].Function.Arguments["host"] != "core" {
		t.Errorf("calls = %+v", calls)
	}
}

func TestToolCall_Malformed(t *testing.T) {
	tests := []struct {
		name string
		buf  string
	}{
		{"no-calls", "\n"},
		{"bad-json", `<ifm|tool_call>{"name": "f", "arguments": {</ifm|tool_call>`},
		{"empty-name", "<ifm|tool_call>\n<ifm|arg_key>k</ifm|arg_key><ifm|arg_value>v</ifm|arg_value></ifm|tool_call>"},
		{"unclosed-value", "<ifm|tool_call>f\n<ifm|arg_key>k</ifm|arg_key><ifm|arg_value>v</ifm|tool_call>"},
		{"duplicate-key", "<ifm|tool_call>f<ifm|arg_key>k</ifm|arg_key><ifm|arg_value>1</ifm|arg_value><ifm|arg_key>k</ifm|arg_key><ifm|arg_value>2</ifm|arg_value></ifm|tool_call>"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := parseK2(tt.buf)
			if len(calls) != 1 || calls[0].Status != 2 || calls[0].Error == "" {
				t.Errorf("calls = %+v, want one failed call", calls)
			}
		})
	}
}

func TestJSONCallName_Streaming(t *testing.T) {
	tests := []struct {
		body string
		want string
		ok   bool
	}{
		{`{"na`, "", false},
		{`{"name": "get_wea`, "", false},
		{`{"name": "get_weather"`, "get_weather", true},
		{`{"name":"a\"b"`, `a"b`, true},
	}
	for _, tt := range tests {
		got, ok := jsonCallName(tt.body)
		if got != tt.want || ok != tt.ok {
			t.Errorf("jsonCallName(%q) = %q, %v; want %q, %v", tt.body, got, ok, tt.want, tt.ok)
		}
	}
}

// =============================================================================
// Params
// =============================================================================

func TestAdjustParams_ReasoningEffort(t *testing.T) {
	tests := []struct {
		in           string
		wantEffort   string
		wantThinking string
	}{
		{"", "", model.ThinkingEnabled},
		{"high", "high", model.ThinkingEnabled},
		{"medium", "medium", model.ThinkingEnabled},
		{"low", "low", model.ThinkingEnabled},
		{"minimal", "low", model.ThinkingEnabled},
		{"none", "", model.ThinkingDisabled},
		{"xhigh", "high", model.ThinkingEnabled},
	}

	for _, tt := range tests {
		got := Parser{}.AdjustParams(model.Params{ReasoningEffort: tt.in, Thinking: model.ThinkingEnabled})
		if got.ReasoningEffort != tt.wantEffort || got.Thinking != tt.wantThinking {
			t.Errorf("AdjustParams(%q) = effort %q thinking %q; want %q, %q",
				tt.in, got.ReasoningEffort, got.Thinking, tt.wantEffort, tt.wantThinking)
		}
	}
}

// =============================================================================
// Unclosed reasoning
// =============================================================================

// drain runs tokens and the end-of-generation Flush, returning what each
// channel accumulated.
func drain(t *testing.T, tokens []string) map[model.Channel]string {
	t.Helper()
	sm := Parser{}.NewStateMachine().(*stateMachine)
	out := map[model.Channel]string{}
	for _, token := range tokens {
		got, eog := sm.Classify(token)
		out[got.Channel] += got.Content
		if eog {
			break
		}
	}
	for got := sm.Flush(); got != (model.Result{}); got = sm.Flush() {
		out[got.Channel] += got.Content
	}
	return out
}

func TestUnclosedReasoning_EndOfTurnIsAnswer(t *testing.T) {
	// After a tool result the model often answers without closing the
	// reasoning block the template opened.
	out := drain(t, []string{"<ifm|think>", "Sunny,", " 24C.", "<|ifm|im_end|>"})
	if out[model.ChannelAnswer] != "Sunny, 24C." || out[model.ChannelReasoning] != "" {
		t.Errorf("channels = %#v", out)
	}
}

func TestUnclosedReasoning_VocabEOGIsAnswer(t *testing.T) {
	sm := Parser{}.NewStateMachine().(*stateMachine)
	sm.Classify("<ifm|think>")
	sm.Classify("Sunny.")
	sm.ConsumeVocabEOG("<|ifm|im_end|>")
	if got := sm.Flush(); got.Channel != model.ChannelAnswer || got.Content != "Sunny." {
		t.Errorf("Flush = %+v", got)
	}
}

func TestUnclosedReasoning_TruncatedStaysReasoning(t *testing.T) {
	// No end of turn: generation was cut short (max_tokens).
	out := drain(t, []string{"<ifm|think>", "Let me", " consider"})
	if out[model.ChannelReasoning] != "Let me consider" || out[model.ChannelAnswer] != "" {
		t.Errorf("channels = %#v", out)
	}
}

func TestUnclosedReasoning_ToolBlockEndsReasoning(t *testing.T) {
	for _, opener := range []string{"<ifm|tool_calls>", "<ifm|tool_call>"} {
		tokens := []string{"<ifm|think>", "Need weather.", opener}
		if opener == "<ifm|tool_calls>" {
			tokens = append(tokens, "\n", "<ifm|tool_call>")
		}
		tokens = append(tokens, `{"name": "get_weather", "arguments": {"city": "Paris"}}`, "</ifm|tool_call>", "<|ifm|im_end|>")

		out := drain(t, tokens)
		if out[model.ChannelReasoning] != "Need weather." {
			t.Errorf("%s: reasoning = %q", opener, out[model.ChannelReasoning])
		}
		calls := parseK2(out[model.ChannelTool])
		if len(calls) != 1 || calls[0].Status != 0 || calls[0].Function.Name != "get_weather" {
			t.Errorf("%s: tool buffer %q parsed to %+v", opener, out[model.ChannelTool], calls)
		}
	}
}

// =============================================================================
// Fragmented input
// =============================================================================

// channels feeds pieces through a fresh state machine plus the end-of-
// generation Flush, returning each channel's accumulated text.
func channels(pieces []string) map[model.Channel]string {
	sm := Parser{}.NewStateMachine().(*stateMachine)
	out := map[model.Channel]string{}
	for _, piece := range pieces {
		got, eog := sm.Classify(piece)
		out[got.Channel] += got.Content
		if eog {
			break
		}
	}
	for got := sm.Flush(); got != (model.Result{}); got = sm.Flush() {
		out[got.Channel] += got.Content
	}
	for ch, text := range out {
		if ch == model.ChannelNone || text == "" {
			delete(out, ch)
		}
	}
	return out
}

func splitEvery(s string, n int) []string {
	var pieces []string
	for len(s) > n {
		pieces = append(pieces, s[:n])
		s = s[n:]
	}
	return append(pieces, s)
}

func TestFragments_EquivalentAcrossBoundaries(t *testing.T) {
	outputs := map[string]string{
		"reasoning-then-answer": "<ifm|think>Check a < b.</ifm|think>\nYes, a < b.<|ifm|im_end|>",
		"medium-effort":         "<ifm|think_fast>quick</ifm|think_fast>\ndone<|ifm|im_end|>",
		"tool-json": "<ifm|think>Need weather.</ifm|think>\n<ifm|tool_calls>\n" +
			`<ifm|tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</ifm|tool_call>` +
			"\n</ifm|tool_calls><|ifm|im_end|>",
		"tool-xml": "<ifm|think>Search.</ifm|think>\n<ifm|tool_calls>\n<ifm|tool_call>search\n" +
			"<ifm|arg_key>q</ifm|arg_key>\n<ifm|arg_value>a <b></ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls><|ifm|im_end|>",
		"unclosed-answer": "<ifm|think>Sunny, 24C.<|ifm|im_end|>",
	}

	for name, output := range outputs {
		t.Run(name, func(t *testing.T) {
			whole := channels([]string{output})
			for _, n := range []int{1, 2, 3, 5, 7} {
				got := channels(splitEvery(output, n))
				if len(got) != len(whole) {
					t.Fatalf("split %d: channels %#v, whole %#v", n, got, whole)
				}
				for ch, text := range whole {
					if got[ch] != text {
						t.Errorf("split %d: channel %v = %q, whole = %q", n, ch, got[ch], text)
					}
				}
			}

			if tool := whole[model.ChannelTool]; tool != "" {
				calls := parseK2(tool)
				if len(calls) != 1 || calls[0].Status != 0 {
					t.Errorf("tool buffer %q parsed to %+v", tool, calls)
				}
			}
		})
	}
}

func TestFragments_Expected(t *testing.T) {
	got := channels(splitEvery("<ifm|think>Check a < b.</ifm|think>\nYes, a < b.<|ifm|im_end|>", 3))
	if got[model.ChannelReasoning] != "Check a < b." || got[model.ChannelAnswer] != "\nYes, a < b." {
		t.Errorf("channels = %#v", got)
	}

	got = channels(splitEvery("<ifm|think>Sunny, 24C.<|ifm|im_end|>", 4))
	if got[model.ChannelAnswer] != "Sunny, 24C." || got[model.ChannelReasoning] != "" {
		t.Errorf("unclosed channels = %#v", got)
	}
}

func TestFragments_SeveralTagsInOnePiece(t *testing.T) {
	sm := Parser{}.NewStateMachine()
	runSteps(t, "one-piece", sm, []step{
		{"<ifm|think>plan</ifm|think>\nanswer", model.ChannelReasoning, "plan", false},
	})
	if got := sm.(*stateMachine).Flush(); got.Channel != model.ChannelAnswer || got.Content != "\nanswer" {
		t.Errorf("Flush = %+v, want the answer queued behind the reasoning", got)
	}
}

func TestFragments_PartialTagAtEnd(t *testing.T) {
	// A trailing "<ifm|th" never became a tag: it is ordinary answer text.
	got := channels([]string{"x < y and <ifm|th"})
	if got[model.ChannelAnswer] != "x < y and <ifm|th" {
		t.Errorf("channels = %#v", got)
	}
}
