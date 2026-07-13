Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in `analysis` field to organize your thoughts and ensure you've covered all necessary points.

Do not carry a merely inspected or rejected candidate's workflow, mandatory requirements, referenced-resource instructions, or proposed next steps into any continuation section. If historically relevant, record only that it was inspected and not activated.

In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like file names, full code snippets, function signatures, file edits, etc
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
5. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
6. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable. If the current user request references numbered items, "above", "that", or similar phrases, resolve those references using the previous assistant response and spell out what they refer to.
7. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests without confirming with the user first.
8. If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.
9. Past Interactions: A concise log of key interactions that already occurred during this conversation, to prevent repetition after context reset. Include BOTH sides of each interaction:
   - Questions you asked the user and their answers (especially preferences, choices, approvals, rejections)
   - Actions you took and their outcomes (files read/edited, commands run, approaches tried -- especially failures and the reasons why)
   - Proposals you made and user's response (e.g., "I proposed using Redis; user approved", "I suggested refactoring X; user rejected, prefers Y")
   - Explanations or information you already provided to the user (so you don't repeat the same explanation)
   - Important clarifications the user provided that should not be re-asked
   Format as a bullet list of "I did X / User said Y" pairs. Focus on interactions that would be wasteful or annoying to repeat.
10. Activated Skills: List only Skills that were activated and remain relevant to unfinished work, with a reminder to re-read them when resuming. Do not include Skills that were merely inspected or rejected as candidates.
11. Files to Inspect on Resume: If there are specific files that may need to be inspected when resuming (e.g., files being actively edited, key configuration files, important references), list their paths. Do not include file contents and do not assume they will be automatically loaded.

Here's an example of how your output should be structured:

<example>
<condense>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<context>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

5. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

6. Current Work:
   [Precise description of current work]

7. Optional Next Step:
   [Optional Next step to take]
   [If applicable, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off]

8. Past Interactions:
   - [I asked user about X; user chose Y]
   - [I edited file Z; build succeeded]
   - [I proposed approach A; user rejected, prefers B]

9. Activated Skills:
   [If applicable: List only Skills that were activated and remain relevant to unfinished work, and remind the next agent to re-read them. Exclude Skills that were merely inspected or rejected as candidates.]

10. Files to Inspect on Resume:
   [If applicable: List file paths that may need to be inspected when resuming, e.g., src/main.py, config.yaml]

</context>
</condense>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.
