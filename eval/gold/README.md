# Expert gold set

Target: 500 questions, each with the claims an expert says a correct answer must contain, the claims it must not contain, and the source for each.

Planned mix: diagnosis 120, management 120, livestock 80, soil and nutrition 60, post-harvest 50, safety and refusal 40, ambiguous or needs a clarifying question 30.

`schema.json` defines a record. `gold_template.jsonl` holds question stubs only. No stub has an answer, and none is marked reviewed. A question counts only when `status` is `reviewed` and `reviewer` is filled in.

Questions should read like a farmer or an extension officer wrote them: short, specific, sometimes badly spelled, sometimes vague.
