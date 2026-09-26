package model

import "testing"

func TestPromptReasoningOpener(t *testing.T) {
	tests := []struct {
		prompt string
		want   string
		ok     bool
	}{
		{"<|im_start|>assistant\n<think>", "<think>", true},
		{"<|im_start|>assistant<|open|>think<|sep|>", "<|open|>think<|sep|>", true},
		{"<|ifm|im_start|>assistant\n<ifm|think>", "<ifm|think>", true},
		{"<|ifm|im_start|>assistant\n<ifm|think_fast>", "<ifm|think_fast>", true},
		{"<|ifm|im_start|>assistant\n<ifm|think_faster>", "<ifm|think_faster>", true},
		{"<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>", "", false},
		{"<|im_start|>assistant", "", false},
	}

	for _, tt := range tests {
		got, ok := promptReasoningOpener(tt.prompt)
		if got != tt.want || ok != tt.ok {
			t.Errorf("promptReasoningOpener(%q) = %q, %v; want %q, %v", tt.prompt, got, ok, tt.want, tt.ok)
		}
	}
}
