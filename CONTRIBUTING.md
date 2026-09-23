# Contributing

Small, reproducible improvements are welcome. Open an issue with the task, expected behavior, actual result, machine, and steps to reproduce. Remove credentials and private customer data before sharing logs.

For a new model or dataset, work in a separate experiment folder. Record its source, license, exact version, label rules, split method and test scope. Do not overwrite frozen checkpoints or historical results. Do not call development improvements a final-test win.

For code changes, run the checks relevant to the change. A frontend change should build and preserve the persona interface. Decision-rule changes need tests for uncertain evidence and negative cases. Changes to the public sample must preserve its recorded-output labeling.

Submit a pull request explaining the problem, the change, how you checked it, and any remaining limitation. Contributions to original project code are under the repository's MIT license; do not contribute material you cannot license.
