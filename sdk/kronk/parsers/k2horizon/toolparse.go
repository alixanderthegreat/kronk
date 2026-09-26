package k2horizon

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

const (
	argKeyClose    = "</ifm|arg_key>"
	argTypeOpen    = "<ifm|arg_type>"
	argTypeClose   = "</ifm|arg_type>"
	argValueOpen   = "<ifm|arg_value>"
	argValueClose  = "</ifm|arg_value>"
	jsonNameMarker = `"name"`
)

// parseK2 parses every <ifm|tool_call> in an accumulated tool-call block.
// An unterminated final call (generation cut short) is parsed as far as it
// goes. Each malformed call becomes a failed entry carrying the raw block.
func parseK2(buf string) []model.ResponseToolCall {
	var toolCalls []model.ResponseToolCall

	rest := buf
	for {
		start := strings.Index(rest, toolCallOpen)
		if start == -1 {
			break
		}
		rest = rest[start+len(toolCallOpen):]

		body := rest
		if end := strings.Index(rest, toolCallClose); end != -1 {
			body = rest[:end]
			rest = rest[end+len(toolCallClose):]
		} else {
			rest = ""
		}

		call, err := parseCall(body)
		if err != nil {
			toolCalls = append(toolCalls, failedToolCall(buf, err))
			continue
		}
		toolCalls = append(toolCalls, call)
	}

	if len(toolCalls) == 0 {
		return []model.ResponseToolCall{failedToolCall(buf, errors.New("parse k2-horizon: no tool calls"))}
	}

	return toolCalls
}

func parseCall(body string) (model.ResponseToolCall, error) {
	body = strings.TrimSpace(body)
	if strings.HasPrefix(body, "{") {
		return parseJSONCall(body)
	}
	return parseXMLCall(body)
}

// parseJSONCall parses {"name": "f", "arguments": {…}}. Arguments encoded
// as a JSON string holding an object are unwrapped.
func parseJSONCall(body string) (model.ResponseToolCall, error) {
	var envelope struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(body), &envelope); err != nil {
		return model.ResponseToolCall{}, fmt.Errorf("parse k2-horizon json: %w", err)
	}

	name := strings.TrimSpace(envelope.Name)
	if name == "" {
		return model.ResponseToolCall{}, errors.New("parse k2-horizon json: function name is empty")
	}

	rawArgs := bytes.TrimSpace(envelope.Arguments)
	if len(rawArgs) > 0 && rawArgs[0] == '"' {
		var inner string
		if err := json.Unmarshal(rawArgs, &inner); err != nil {
			return model.ResponseToolCall{}, fmt.Errorf("parse k2-horizon json: arguments of %q: %w", name, err)
		}
		rawArgs = []byte(inner)
	}

	args := model.ToolCallArguments{}
	if len(rawArgs) > 0 && string(rawArgs) != "null" {
		decoded, ok := decodeJSONValue(string(rawArgs))
		object, isObject := decoded.(map[string]any)
		if !ok || !isObject {
			return model.ResponseToolCall{}, fmt.Errorf("parse k2-horizon json: arguments of %q are not an object", name)
		}
		args = object
	}

	return newToolCall(name, args), nil
}

// parseXMLCall parses the xml and xml_typed formats:
//
//	NAME
//	<ifm|arg_key>K</ifm|arg_key>
//	[<ifm|arg_type>T</ifm|arg_type>]
//	<ifm|arg_value>V</ifm|arg_value>
func parseXMLCall(body string) (model.ResponseToolCall, error) {
	nameEnd := len(body)
	if at := strings.Index(body, argKeyOpen); at != -1 {
		nameEnd = at
	}
	name := strings.TrimSpace(body[:nameEnd])
	if name == "" {
		return model.ResponseToolCall{}, errors.New("parse k2-horizon xml: function name is empty")
	}

	args := model.ToolCallArguments{}
	rest := strings.TrimSpace(body[nameEnd:])
	for rest != "" {
		key, after, err := cutTag(rest, argKeyOpen, argKeyClose)
		if err != nil {
			return model.ResponseToolCall{}, fmt.Errorf("parse k2-horizon xml: function %q: %w", name, err)
		}
		if key == "" {
			return model.ResponseToolCall{}, fmt.Errorf("parse k2-horizon xml: function %q: empty argument key", name)
		}
		if _, dup := args[key]; dup {
			return model.ResponseToolCall{}, fmt.Errorf("parse k2-horizon xml: function %q: argument %q is duplicated", name, key)
		}
		rest = strings.TrimSpace(after)

		argType := ""
		if strings.HasPrefix(rest, argTypeOpen) {
			argType, after, err = cutTag(rest, argTypeOpen, argTypeClose)
			if err != nil {
				return model.ResponseToolCall{}, fmt.Errorf("parse k2-horizon xml: function %q argument %q: %w", name, key, err)
			}
			rest = strings.TrimSpace(after)
		}

		value, after, err := cutTag(rest, argValueOpen, argValueClose)
		if err != nil {
			return model.ResponseToolCall{}, fmt.Errorf("parse k2-horizon xml: function %q argument %q: %w", name, key, err)
		}
		rest = strings.TrimSpace(after)

		args[key] = xmlValue(value, strings.TrimSpace(argType))
	}

	return newToolCall(name, args), nil
}

// cutTag expects s to start with open, and returns the text up to close and
// everything after it.
func cutTag(s, open, close string) (string, string, error) {
	if !strings.HasPrefix(s, open) {
		return "", "", fmt.Errorf("expected %s", open)
	}
	s = s[len(open):]

	end := strings.Index(s, close)
	if end == -1 {
		return "", "", fmt.Errorf("%s is not closed", open)
	}

	return s[:end], s[end+len(close):], nil
}

// xmlValue converts an xml argument value. With an explicit xml_typed type
// the value is converted to it; without one, JSON arrays and objects are
// decoded (the template writes them as JSON literals) and everything else
// stays text for ToolCallWithSchema to type from the tool declaration.
func xmlValue(raw, argType string) any {
	if argType != "" {
		if converted, ok := convertToType(raw, argType); ok {
			return converted
		}
		return raw
	}

	trimmed := strings.TrimSpace(raw)
	if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
		if decoded, ok := decodeJSONValue(trimmed); ok {
			return decoded
		}
	}

	return raw
}

// convertToType converts text to a JSON-schema type. It reports false when
// the text does not fit the type, leaving the caller to keep the string.
func convertToType(raw, schemaType string) (any, bool) {
	trimmed := strings.TrimSpace(raw)

	switch strings.ToLower(schemaType) {
	case "string", "str":
		return raw, true

	case "boolean", "bool":
		switch strings.ToLower(trimmed) {
		case "true":
			return true, true
		case "false":
			return false, true
		}

	case "integer", "int":
		if number, ok := decodeNumber(trimmed); ok && isJSONInteger(number) {
			return number, true
		}

	case "number", "float":
		if number, ok := decodeNumber(trimmed); ok {
			return number, true
		}

	case "object", "dict":
		if decoded, ok := decodeJSONValue(trimmed); ok {
			if _, isObject := decoded.(map[string]any); isObject {
				return decoded, true
			}
		}

	case "array", "list":
		if decoded, ok := decodeJSONValue(trimmed); ok {
			if _, isArray := decoded.([]any); isArray {
				return decoded, true
			}
		}

	case "null", "none":
		if trimmed == "null" || trimmed == "" {
			return nil, true
		}
	}

	return nil, false
}

// normalizeArguments types xml-format string arguments using the matching
// function's declared schema. Arguments without an unambiguous declared
// type, and non-string arguments, are left unchanged.
func normalizeArguments(toolCalls []model.ResponseToolCall, tools []model.D) {
	for i := range toolCalls {
		properties := toolProperties(tools, toolCalls[i].Function.Name)
		if properties == nil {
			continue
		}

		for key, value := range toolCalls[i].Function.Arguments {
			raw, ok := value.(string)
			if !ok {
				continue
			}

			property, ok := properties[key].(model.D)
			if !ok {
				continue
			}

			schemaType, ok := property["type"].(string)
			if !ok || schemaType == "string" {
				continue
			}

			if converted, ok := convertToType(raw, schemaType); ok {
				toolCalls[i].Function.Arguments[key] = converted
			}
		}
	}
}

// toolProperties returns the declared parameter properties of the named
// function, or nil when it is undeclared or declared more than once.
func toolProperties(tools []model.D, name string) model.D {
	var properties model.D

	for _, tool := range tools {
		function, ok := tool["function"].(model.D)
		if !ok || function["name"] != name {
			continue
		}
		if properties != nil {
			return nil
		}

		parameters, ok := function["parameters"].(model.D)
		if !ok {
			return nil
		}
		if properties, ok = parameters["properties"].(model.D); !ok {
			return nil
		}
	}

	return properties
}

// jsonCallName extracts the "name" value from a possibly incomplete JSON
// call body, reporting false until the whole string value has arrived.
func jsonCallName(body string) (string, bool) {
	_, after, ok := strings.Cut(body, jsonNameMarker)
	if !ok {
		return "", false
	}

	rest := strings.TrimLeft(after, " \t\r\n")
	if !strings.HasPrefix(rest, ":") {
		return "", false
	}
	rest = strings.TrimLeft(rest[1:], " \t\r\n")
	if !strings.HasPrefix(rest, `"`) {
		return "", false
	}

	escaped := false
	for i := 1; i < len(rest); i++ {
		switch {
		case escaped:
			escaped = false
		case rest[i] == '\\':
			escaped = true
		case rest[i] == '"':
			var name string
			if err := json.Unmarshal([]byte(rest[:i+1]), &name); err != nil {
				return "", false
			}
			name = strings.TrimSpace(name)
			return name, name != ""
		}
	}

	return "", false
}

func decodeJSONValue(raw string) (any, bool) {
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.UseNumber()

	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, false
	}
	if decoder.More() {
		return nil, false
	}

	return value, true
}

func decodeNumber(raw string) (json.Number, bool) {
	value, ok := decodeJSONValue(raw)
	if !ok {
		return "", false
	}
	number, ok := value.(json.Number)
	return number, ok
}

func isJSONInteger(number json.Number) bool {
	_, err := number.Int64()
	return err == nil
}

func newToolCall(name string, args model.ToolCallArguments) model.ResponseToolCall {
	return model.ResponseToolCall{
		ID:   newToolCallID(),
		Type: "function",
		Function: model.ResponseToolCallFunction{
			Name:      name,
			Arguments: args,
		},
	}
}

func failedToolCall(raw string, err error) model.ResponseToolCall {
	return model.ResponseToolCall{
		ID:     newToolCallID(),
		Type:   "function",
		Status: 2,
		Raw:    raw,
		Error:  err.Error(),
	}
}
