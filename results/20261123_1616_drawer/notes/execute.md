Operator note 19:19: the loop works now; skip separate real test runs between episodes unless a change is risky
(the next episode is the test, and each extra run costs a reset and ~5 min of the RL budget). Keep changes to the
scripted part small and rare so the learned-vs-scripted comparison stays clean.
