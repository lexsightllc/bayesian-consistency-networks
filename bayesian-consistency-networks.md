Bayesian Consistency Networks: A Probabilistic Truth Detector
What This Project Does
This system resolves conflicting claims from multiple sources by evaluating source reliability and applying logical consistency rules. Instead of black-and-white answers, it provides probability scores for each claim's truthfulness, acting as a "probabilistic referee" that considers evidence, reputation, and logical rules.

Core Components
Propositions (Claims)
Binary statements that can be true or false
Start with a neutral 50% probability
Get updated based on evidence and logical rules
Sources
People, tools, or databases that provide information
Each source has two learned reliability metrics:
Sensitivity: Accuracy when the truth is "yes" (true positives)
Specificity: Accuracy when the truth is "no" (true negatives)
Soft Logical Constraints
Exclusion: Two claims can't both be true
Implication (A ⇒ B): If A is true, B can't be false
Equivalence (A ⇔ B): A and B must agree (both true or both false)
These are "soft" constraints - they penalize but don't strictly forbid inconsistencies, making the system robust to noise and exceptions.
How It Works
Each source's vote pushes probabilities up or down based on their reliability
Logical constraints apply additional pressure against inconsistent combinations
The system learns which sources are more reliable over time
This process repeats until beliefs stabilize
Key Features
Handles noisy, conflicting information
Learns source reliability automatically
Enforces logical consistency
Provides explainable results showing why each conclusion was reached
Numerically stable and robust
Practical Applications
Fact-checking systems
Decision support tools
Information aggregation from multiple AI agents
Any scenario requiring consistent truth assessment from conflicting sources
Technical Implementation
The system is implemented in Python with:

Clean, modular design
Numerical stability safeguards
Efficient belief propagation
Extensible architecture for new constraint types
The core algorithm balances evidence from sources with the need for logical consistency, learning which sources to trust while maintaining coherent beliefs across all propositions.