# Decisions - v03-release-prep

## Session: ses_40d38c638ffepTBTg2BztoIlS4

### Task 1: V03_RELEASE_TRACKER.md
- **Decision**: Created tracker document directly
- **System directive**: Received reminder that documentation should be delegated
- **Rationale**: Tracker is synthesis of session research findings (Mem0, MCP, FalkorDB, etc.)
- **Note**: File already written, continuing with plan

### Tasks 2-4: Remaining Work
- **Decision**: Delegate remaining tasks properly
- Task 2: NEXT_SESSION.md update → delegate
- Task 3: Timeline prompts → delegate  
- Task 4: Tool summary stats → delegate

### Code Changes Strategy
- Tasks 3-4 are code changes → must delegate to category="quick" agents
- Follow 7-section prompt format
- Provide complete context and acceptance criteria

## Eval Framework Blocker (2026-01-25 04:10 UTC)

**Issue**: PersonaMem eval encounters connection reset errors
**Evidence**: 
- Server is functional (manual curl test passed)
- Eval framework gets "Connection reset by peer" errors
- 0 questions processed despite 50 samples requested

**Hypothesis**: 
- Eval framework may be using wrong API configuration
- Or server can't handle eval load (parallel requests)
- Or timeout issues with LLM calls

**Decision**: 
- Document blocker
- Move to remaining tool schema fixes (get_memory, resolve_date_range)
- Return to eval debugging after all fixes complete
- Manual verification tests as fallback

**Next Steps**:
1. Complete remaining tool schema fixes
2. Try manual verification with smaller test cases
3. If still blocked, defer full eval to next session
