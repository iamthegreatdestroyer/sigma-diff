# Phase 3 Communication Guidelines & Standards

**Version**: 1.0  
**Effective Date**: January 6, 2026  
**Owner**: @MENTOR  
**Audience**: All Phase 3 team members

---

## 📋 Communication Channels & Their Purpose

### 1. GitHub Issues - Task-Level Discussion

**Purpose**: Work coordination, design decisions, technical problem-solving  
**Audience**: Relevant team members + async stakeholders  
**Response Time**: <24 hours for active discussions

**When to use**:

- ✓ Breaking down sprint work into issues
- ✓ Design discussions and alternatives
- ✓ Code review feedback
- ✓ Bug reports and fixes
- ✓ Documentation updates
- ✓ Async questions that need context

**When NOT to use**:

- ✗ Quick status updates (use Teams for that)
- ✗ Urgent escalations (call instead)
- ✗ Private matters (use direct message)
- ✗ Social discussion (use #general channel)

**GitHub Etiquette**:

```markdown
# Issue Title Format

For Sprint work:

- "feat: [Sprint] [Component] [Feature description]"
- "fix: [Sprint] [Component] [Bug description]"
- "docs: [Component] [Documentation update]"
- "perf: [Component] [Performance improvement]"

Example:

- "feat: Sprint-1.1 distributed inference tensor parallel layer"
- "fix: Sprint-1.2 batch inference dynamic batching edge case"
- "docs: Sprint-2.1 REST API endpoint documentation"
```

**Issue Labels**:

```
Type: feature, bug, documentation, performance, testing
Status: open, in-progress, blocked, review, done
Sprint: 1.1, 1.2, 2.1, 2.2
Priority: p0-critical, p1-high, p2-medium, p3-low
Component: inference, distributed, api, monitoring, infrastructure
```

**Issue Description Template**:

```markdown
## Problem/Feature

[Clear 1-2 sentence description]

## Context

[Why this matters, related issues]

## Proposed Solution

[How to solve it]

## Acceptance Criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Dependencies

[Other issues/tasks needed first]

## Estimate

[Story points: 1, 2, 3, 5, 8]
```

---

### 2. GitHub Projects - Sprint Tracking & Status

**Purpose**: Visual sprint status, velocity tracking, roadmap  
**Audience**: Entire Phase 3 team  
**Update Frequency**: Daily (when working on issues)

**When to use**:

- ✓ Tracking sprint progress
- ✓ Updating issue status (moving between columns)
- ✓ Checking sprint velocity
- ✓ Weekly report generation
- ✓ Identifying blockers (use labels)

**GitHub Projects Board States**:

```
BACKLOG → TODO → IN-PROGRESS → IN-REVIEW → DONE

BACKLOG
  - Issues not yet scheduled
  - Future enhancement ideas
  - Tech debt items

TODO
  - Committed to current sprint
  - Ready to start
  - No dependencies

IN-PROGRESS
  - Currently being worked on
  - Someone assigned
  - Update daily in standup

IN-REVIEW
  - Code review pending
  - Design review pending
  - Testing pending

DONE
  - Issue closed
  - Acceptance criteria met
  - Tests passing
```

**Best Practices**:

- Move issue to IN-PROGRESS when starting work
- Add comment with status updates if issue stalls
- Link related issues with GitHub issue linking syntax
- Tag blockers immediately with 🚨 emoji in comment

---

### 3. Microsoft Teams / Slack - Daily Coordination

**Purpose**: Real-time chat, daily standup, quick decisions  
**Audience**: Team + availability for async response  
**Response Time**: <15 minutes for standup messages, <1 hour for general chat

**Channels**:

```
#ryzen-llm-phase3              - General phase 3 discussion
#ryzen-llm-phase3-standup      - Daily standup notes (threaded)
#ryzen-llm-phase3-checkpoint   - Wednesday mid-sprint updates
#ryzen-llm-phase3-review       - Friday sprint reviews
#ryzen-llm-phase3-retro        - Friday retrospective notes
#ryzen-llm-blockers            - Real-time blocker escalation (🚨)
#ryzen-llm-performance         - Performance metrics discussion
#ryzen-llm-infrastructure      - GPU/infra status updates
```

**When to use Teams**:

- ✓ Quick status updates ("Almost done with X")
- ✓ Standup answers (threaded in daily standup post)
- ✓ Fast decision requests ("Can we do X or Y?")
- ✓ Blocker escalation (tag @ARCHITECT in #ryzen-llm-blockers)
- ✓ Real-time coordination during active work
- ✓ Social team building & celebration

**When NOT to use Teams**:

- ✗ Detailed technical design (use GitHub Issues)
- ✗ Decision documentation (use GitHub + Decision Log)
- ✗ Permanent decisions (easily forgotten in chat)

**Teams Best Practices**:

- Use **threads** to keep discussions organized
- Mention `@channel` only for critical blockers
- Use emoji reactions (+1, -1, 👍, etc.) for quick feedback
- Pin important decisions/info in channel header
- Archive sprint channel after completion (reference only)

**Daily Standup Format in Teams**:

```
# Standup - Tuesday, January 7, 2025

**Lead**: @APEX

---

### 📋 Who's Done What?

**@developer1**:
└─ ✅ Completed tensor parallel layer implementation
└─ ✅ 150 unit tests passing

**@developer2**:
└─ ✅ NCCL communication benchmark complete

### 🎯 Who's Doing What?

**@developer1**:
└─ 🔨 Integrating tensor parallel layer into model loader
└─ 📆 Target: Complete by EOD Wednesday

**@developer2**:
└─ 🔨 Tuning NCCL parameters for optimal throughput
└─ 📆 Target: Complete by Thursday

### 🚨 Blockers?

**@developer3**:
└─ 🛑 BLOCKER: GPU quota issue - only 2 of 4 GPUs available
└─ 👤 Owner: @FLUX
└─ ⏱️ ETA: Fixed by EOD today
└─ 💬 GitHub Issue: #234

---

**Next standup**: Wednesday 9:15 AM UTC
```

---

### 4. Email - Weekly Summaries & Decisions

**Purpose**: Official record, stakeholder updates, decision communication  
**Audience**: Team + stakeholders + executives  
**Frequency**: Weekly Friday 6:00 PM UTC

**When to use**:

- ✓ Weekly status reports
- ✓ Go/No-Go gate decisions
- ✓ Official scope changes
- ✓ Phase-level milestones
- ✓ Executive updates

**Email Subject Lines**:

```
Phase 3 Weekly Status Report - Week of Jan 6
DECISION: Sprint 1.1 Go/No-Go Gate - Jan 17
ESCALATION REQUIRED: [Issue] - Response needed by [Date]
Phase 3 Progress Update - [Milestone]
```

**Email Template**:

```
To: @ARCHITECT, All leads, Stakeholders
Subject: Phase 3 Weekly Status - Week of [Date]

---

## Executive Summary
[One paragraph: What happened this week, overall status]

## Sprint Progress

### Sprint 1.1 [if active]
- Status: ON-TRACK / AT-RISK / COMPLETE
- Completed: 12 of 15 committed issues
- Velocity: 45 story points

### Sprint 1.2 [if active]
- Status: ON-TRACK / AT-RISK
- Completed: 8 of 16 committed issues
- Velocity: 35 story points

### Key Metrics
- Test coverage: 92% (target: >90%)
- Performance vs target: 3.9x (target: 3.8-4.2x) ✅
- Blockers: 2 open (both <4 hours old)

## Top 3 Blockers
1. GPU quota increased yesterday - RESOLVED
2. NCCL optimization requires tuning - @VELOCITY owns, ETA Jan 10
3. API design decision pending - Decision meeting Jan 9

## Risks & Mitigations
- Risk: Performance plateau at 3.9x speedup
  Mitigation: @VELOCITY optimization work, priority shifted

## Decisions Made
- ✅ Approved: Batch size optimization (decision log entry #3)
- ✅ Approved: Infrastructure upgrade for benchmarking

## Next Week
1. Complete Week 2 performance validation
2. Finalize Sprint 1.1 Go/No-Go gate (Jan 17, 4 PM)
3. Begin Sprint 1.2 planning

## Questions or Concerns?
Reply-all to discuss or ping @ARCHITECT directly.

---
@ARCHITECT
```

---

## 🚨 Escalation Triggers & Communication

### Critical Blocker Escalation Path

**Step 1: Team Awareness (Immediate)**

```
Post in #ryzen-llm-blockers:
"🚨 BLOCKER: [Brief description]
Sprint impact: [What can't proceed]
Owner: @[Responsible person]
Escalation: @ARCHITECT
ETA to resolve: [Best estimate]"
```

**Step 2: Incident Channel (If >30 min critical)**

```
Create incident channel: #incident-[YYYYMMDD]-[name]
- Invite: @ARCHITECT + affected team
- Purpose: Real-time collaboration to resolve
- Standup: 15-min call if >1 hour critical
```

**Step 3: Executive Alert (If >2 hour impact)**

```
Email with subject: "CRITICAL BLOCKER - Phase 3 Execution Risk"
- To: @ARCHITECT, Project Sponsor
- Include: Impact, ETA, escalation
- Frequency: Update every 30 minutes
```

**Blocker Resolution SLA**:
| Severity | Response | Resolution |
|----------|----------|------------|
| Critical (blocks >1 person) | <15 min | <2 hours |
| High (blocks 1 person) | <1 hour | <4 hours |
| Medium (slows work) | <4 hours | <1 day |
| Low (nice-to-have) | <1 day | <1 week |

---

## 📧 Async vs Sync Decision Framework

**Use SYNC (Standup/Call) for**:

- Brainstorming new solutions
- Complex architecture decisions
- Tight deadline coordination
- Team morale & retro discussions
- Conflict resolution discussions

**Use ASYNC (GitHub/Email) for**:

- Detailed technical documentation
- Decision records and ADRs
- Code review feedback
- Progress updates
- Issues requiring <24hr response

**Decision Matrix**:

```
Urgency? HIGH → Sync (standup/call)
         LOW  → Async (GitHub/email)

Complexity? HIGH → Async first (document fully), sync if needed
            LOW  → Can use either

Documentation Needed? YES → Async (GitHub/email)
                    NO  → Sync OK
```

---

## 📝 Documentation Standards

### Issue Documentation

- Clear, specific title (not "Fix bugs")
- Context section explaining "why"
- Acceptance criteria (testable)
- Links to related issues/PRs

### Code Comments

- Why code exists (not what it does)
- Non-obvious logic explained
- References to GitHub issues
- Example: `// Distribute tensors row-wise (see #234 for perf analysis)`

### Commit Messages

```
feat: Sprint-1.1 distributed tensor parallelism layer

- Implement DistributedTensorLayer with row-wise distribution
- Add NCCL communication primitives
- 45 unit tests, 90% coverage
- Perf target: 3.8x speedup on 4 GPUs

Fixes #234, Relates to #456
```

### Pull Request (Code Review)

```markdown
## What

[What changes were made]

## Why

[Why these changes were needed]

## How

[How the solution works]

## Testing

[How testing was done]

## Impact

[Performance, scope, dependencies]

## Checklist

- [ ] Tests passing
- [ ] Coverage >90%
- [ ] No regressions on Phase 2
- [ ] Documentation updated
```

---

## 🤝 Code Review Expectations

**Response Time**: <24 hours for active PR

**Review Checklist**:

- [ ] Code accomplishes stated goal
- [ ] No obvious bugs or edge cases
- [ ] Follows project code style
- [ ] Tests provided and passing
- [ ] Documentation updated
- [ ] No regressions on main branch
- [ ] Performance acceptable (benchmark if needed)

**Review Comment Styles**:

```markdown
# Request Changes (blocks merge)

🛑 **Request changes**: This logic doesn't handle case X.
Suggestion: Add condition for X (see line YZ)

# Suggest Improvement (non-blocking)

💡 **Suggestion**: Could optimize this loop with vectorization.
Not required for this PR but would help Sprint 1.2.

# Approval

✅ **Approved**: Looks good, well tested, merges cleanly.
```

---

## 📊 Weekly Metrics Communication

**Reported in**: Friday sprint review + weekly email

**Key Metrics**:

```markdown
## Performance Metrics

- Distributed inference speedup: 3.9x (target: 3.8-4.2x) ✅
- Single GPU baseline: 100 tokens/sec
- 4-GPU actual: 390 tokens/sec

## Quality Metrics

- Test coverage: 92% (target: >90%) ✅
- Code review turn-around: 14 hours avg (target: <24h) ✅
- Regression bugs: 0 (target: 0) ✅

## Team Metrics

- Standup attendance: 96% (target: >95%) ✅
- Blocker resolution time: 2.3 hours avg (target: <4h) ✅
- Team satisfaction: 4.2/5 (target: ≥4.0) ✅
```

---

## 🎯 Decision Communication Template

**When a major decision is made**, communicate with this template:

```markdown
# [DECISION TITLE]

**Date**: [Date and time]  
**Decision Owner**: [@Name]  
**Status**: APPROVED / PENDING FEEDBACK

## What We Decided

[Clear statement of what was decided]

## Why

[Brief rationale - 3-4 sentences max]

## What This Means For You

### If you're on [Component/Team]

- [Impact 1]
- [Impact 2]

### If you're on [Other Component/Team]

- [Impact 1]

## Action Items

- [ ] [Action] - Owner: [@Name] - Due: [Date]
- [ ] [Action] - Owner: [@Name] - Due: [Date]

## Questions?

Reply in GitHub Issue [#XXX] or ask in standup.
```

---

## 🚫 Communication Anti-Patterns to Avoid

| Anti-Pattern                                     | Why Bad                     | What To Do                                   |
| ------------------------------------------------ | --------------------------- | -------------------------------------------- |
| Long Slack threads without resolution            | Hard to find answer later   | Copy final decision to GitHub issue          |
| "URGENT!!1!" without context                     | Creates false urgency       | Provide 30-sec summary of impact             |
| Decisions made in Slack, never documented        | Institutional forgetting    | Create GitHub issue + decision log entry     |
| Vague status updates ("working on it")           | No visibility into progress | Be specific: "Completed Part A, Part B next" |
| No blocker escalation (just complaining in chat) | Blocker stays unresolved    | Post to #ryzen-llm-blockers with @ARCHITECT  |
| Async decision-making on sync-only topics        | Spirals in comment threads  | Call a meeting instead                       |
| Cancelling standups                              | Loss of coordination        | Move to async if truly not needed            |
| Merge without review                             | Regressions                 | Always require code review                   |

---

## ✅ Communication Dos & Don'ts

### ✅ DO:

- Be specific: "Speedup is 3.9x" not "It's working"
- Provide context: Why you're asking/deciding something
- Link related issues: "Relates to #234"
- Escalate early: Don't wait until 11:59 PM on deadline
- Celebrate wins: "Great work team on hitting the 3.8x target!"
- Ask questions: "Is this approach right? See issue #456"
- Document decisions: Even small ones go in GitHub

### 🚫 DON'T:

- Leave blockers unescalated: Report immediately
- Make decisions without team input: Async discuss first
- Forget to update GitHub: Teams chat is not permanent
- Assume everyone knows context: Always explain why
- Merge without tests/review: Code quality over speed
- Ignore deadline warnings: Escalate early if at risk
- Let decisions stay in Slack: Move to issue/decision log

---

## 📞 When to Have a Synchronous Call

**Schedule a call if**:

- Decision needs <4 hours (not enough time for async)
- Conflict between options (people disagree)
- Complex discussion with >2 people involved
- Brainstorming new approaches
- Architecture walkthrough needed

**Standup is appropriate for**:

- Status updates (what's done, doing, blockers)
- Quick escalations
- Quick decisions (<5 min discussion)
- Announcements
- Celebration of wins

**Don't use standup for**:

- Long design discussions (schedule separate call)
- Code review feedback (use GitHub PR comments)
- Detailed problem-solving (unless quick question)
- Retro/morale topics (use Friday retro instead)

---

## 🎓 Team Communication Health Metrics

**Measured weekly, reported in standups**:

| Metric                      | Target    | How Measured                    |
| --------------------------- | --------- | ------------------------------- |
| Issue response time         | <24 hours | GitHub issue comment timestamps |
| PR review turn-around       | <24 hours | GitHub PR review timestamps     |
| Blocker resolution time     | <4 hours  | Time from report to resolution  |
| Email response time         | <4 hours  | Internal email tracking         |
| Standup attendance          | >95%      | Calendar tracking               |
| Decision documentation rate | 100%      | GitHub vs chat decisions        |
| Team satisfaction           | ≥4.0/5    | Weekly pulse survey             |

---

**Phase 3 communication guidelines established. All channels defined, escalation paths clear, etiquette standards set. Team ready for coordinated execution.**
