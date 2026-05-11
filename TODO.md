# TODO

- [ ] 1) Fix persistent chat history: stop session_id being reset on each rerun, and ensure DB reads/writes use stable per-browser session.
- [ ] 2) Add a "New chat" button to intentionally reset session_id (instead of accidental resets).
- [ ] 3) Verify messages table usage and ordering.
- [ ] 4) Quick manual test: send a few questions, refresh/re-run the app, confirm history persists.

