// Package k2horizon implements the Parser for IFM's K2-Horizon models.
//
// K2-Horizon wraps reasoning in one of three effort-specific tag pairs,
// all single tokens in its vocabulary:
//
//	<ifm|think>…</ifm|think>               reasoning_effort=high (default)
//	<ifm|think_fast>…</ifm|think_fast>     reasoning_effort=medium
//	<ifm|think_faster>…</ifm|think_faster> reasoning_effort=low
//
// The chat template opens the reasoning tag in the prompt itself, so
// generation normally begins inside reasoning.
//
// Tool calls sit in a single <ifm|tool_calls>…</ifm|tool_calls> block, one
// <ifm|tool_call>…</ifm|tool_call> per call, in the format selected by the
// tool_call_format chat template kwarg:
//
//	json:      <ifm|tool_call>{"name": "f", "arguments": {…}}</ifm|tool_call>
//	xml:       <ifm|tool_call>f
//	           <ifm|arg_key>k</ifm|arg_key>
//	           <ifm|arg_value>v</ifm|arg_value>
//	           </ifm|tool_call>
//	xml_typed: as xml, with <ifm|arg_type>t</ifm|arg_type> before each value
//
// The xml format is the template default. All three are parsed.
//
// This mirrors the k2_horizon reasoning and tool-call parsers IFM ships for
// vLLM and SGLang.
package k2horizon

import (
	"context"
	"strings"

	"github.com/ardanlabs/kronk/sdk/kronk/applog"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

// name is the canonical name returned by Parser.Name.
const name = "k2-horizon"

// Parser implements model.Parser for K2-Horizon.
type Parser struct{}

// New returns a Parser value if the fingerprint indicates a K2-Horizon
// model, otherwise returns false. Detection is layered: GGUF
// "general.architecture" ("k2-horizon") is the strongest signal, the chat
// template's ifm-namespaced reasoning and tool-call markers are next, and
// the model name substring is a last-resort fallback.
func New(fp model.Fingerprint) (model.Parser, bool) {
	// 1. GGUF architecture.
	arch := strings.ToLower(fp.Architecture)
	if strings.HasPrefix(arch, "k2-horizon") || strings.HasPrefix(arch, "k2_horizon") {
		return Parser{}, true
	}

	// 2. Chat template markers distinctive to K2-Horizon.
	if containsK2Markers(fp.ChatTemplate) {
		return Parser{}, true
	}

	// 3. Model name fallback.
	modelName := strings.ToLower(fp.ModelName)
	if strings.Contains(modelName, "k2-horizon") || strings.Contains(modelName, "k2_horizon") {
		return Parser{}, true
	}

	return Parser{}, false
}

// Name returns the parser identifier.
func (Parser) Name() string { return name }

// NewStateMachine returns a fresh per-slot streaming state machine.
func (Parser) NewStateMachine() model.StateMachine {
	return &stateMachine{status: model.ChannelAnswer}
}

// ToolCall parses the accumulated <ifm|tool_calls> block.
func (Parser) ToolCall(_ context.Context, _ applog.Logger, buf string) []model.ResponseToolCall {
	return parseK2(buf)
}

// ToolCallWithSchema parses tool calls and uses the declared tool schema to
// recover argument types from the untyped xml format, where scalars are
// written as plain text. The json and xml_typed formats already carry types
// and pass through unchanged.
func (Parser) ToolCallWithSchema(_ context.Context, _ applog.Logger, buf string, tools []model.D) []model.ResponseToolCall {
	toolCalls := parseK2(buf)
	normalizeArguments(toolCalls, tools)
	return toolCalls
}

// AdjustParams coerces reasoning_effort into the values K2-Horizon's chat
// template accepts: high, medium and low. Any other value makes the
// template raise and fails the request. "none" turns thinking off (the
// template then renders an empty reasoning block), "minimal" becomes
// "low", and anything else unrecognized becomes "high", the model's
// recommended setting. An empty value remains unset so the template applies
// its native default (high).
func (Parser) AdjustParams(params model.Params) model.Params {
	switch params.ReasoningEffort {
	case "", model.ReasoningEffortHigh, model.ReasoningEffortMedium, model.ReasoningEffortLow:
		// Already valid, or left to the template default.

	case model.ReasoningEffortNone:
		params.Thinking = model.ThinkingDisabled
		params.ReasoningEffort = ""

	case model.ReasoningEffortMinimal:
		params.ReasoningEffort = model.ReasoningEffortLow

	default:
		params.ReasoningEffort = model.ReasoningEffortHigh
	}

	return params
}

// containsK2Markers reports whether a chat template carries K2-Horizon's
// ifm-namespaced reasoning or tool-call tokens.
func containsK2Markers(template string) bool {
	for _, marker := range []string{
		"<ifm|tool_call>",
		"<ifm|think>",
	} {
		if strings.Contains(template, marker) {
			return true
		}
	}
	return false
}
