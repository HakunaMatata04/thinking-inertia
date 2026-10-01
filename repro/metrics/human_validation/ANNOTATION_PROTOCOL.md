# Annotation Protocol

Judge only the supplied **QUESTION** and **PRE-ANSWER TEXT**. Do not infer
hidden reasoning and do not judge whether the final answer is correct.

For every non-empty pre-answer text, verify or revise all three drafted fields.

## A. Question relevance (1--5)

1. **No relevant content.** Empty, generic, off-topic, or unrelated text.
2. **Weak relevance.** Mostly boilerplate or only loosely related to the
   question.
3. **On-topic but non-evidential.** Relevant wording, restatement, definition,
   or assertion without question-specific evidence or an inferential step.
4. **Question-specific support.** At least one piece of question-specific
   evidence, comparison, computation, or inference that helps answer the
   question.
5. **Substantive multi-step support.** Multiple connected, question-specific
   steps or a developed derivation.

The relevance score does not by itself decide whether reasoning is present.
A paraphrase can be relevant without containing inference.

## B. Visible-reasoning label

- `generic_or_off_topic`: generic lead-in, formatting text, or off-topic text
  with no question-specific inference.
- `paraphrase_only`: restates the question or supplied facts without connecting
  them through an inference.
- `relevant_noninferential`: supplies a relevant fact, definition,
  description, or answer assertion without an inferential step.
- `explicit_reasoning`: contains at least one visible deduction, computation,
  comparison, option elimination, rule application, evidence-to-conclusion
  link, or causal step addressing the question. It need not be deep or correct.
- `unclear`: cannot be classified reliably from the visible text.

## C. Supporting span

Record a short exact quotation from the pre-answer text that best supports the
reasoning label. Do not paraphrase it. For `unclear`, record the most relevant
ambiguous span when possible.

## Blinding and review

The model draft is an editable starting point. Annotators must inspect the
question and complete pre-answer text before confirming it. Draft-model
identity and all experimental metadata are hidden. Human exports preserve both
the initial draft and the final decision so draft-change rates can be audited.

