# Phase 3 Standup Calendar & Meeting Schedule

**Effective Date**: January 6, 2026  
**Duration**: 22 weeks through May 3, 2026  
**Facilitation**: @OMNISCIENT (overall coordination)

---

## 📅 Weekly Standup Pattern

### Daily Standup (Monday-Friday)

- **Time**: 9:15 AM - 9:30 AM UTC
- **Duration**: 15 minutes (strict timeboxing)
- **Platform**: Microsoft Teams (primary) | Slack async thread (backup)
- **Facilitator**: Rotating (APEX → VELOCITY → FLUX → SYNAPSE → SENTRY)
- **Attendees**: All Phase 3 team members + tech leads
- **Required**: Yes

**Format (3 Questions)**:

1. What did you complete since last standup?
2. What are you working on before next standup?
3. What blockers do you have?

**Notes Captured In**: GitHub Issues (standup tag)

---

### Mid-Sprint Checkpoint (Wednesday)

- **Time**: 2:00 PM - 2:30 PM UTC
- **Duration**: 30 minutes
- **Day**: Every Wednesday
- **Platform**: Microsoft Teams (required - screen share for metrics)
- **Lead**: ARCHITECT + Sprint lead
- **Attendees**: Component owners + interested parties (optional)
- **Required**: Yes for leads, optional for others

**Agenda**:

1. Progress against sprint plan (5 min)
2. Current blockers & mitigation (10 min)
3. Scope/timeline adjustments (10 min)
4. Resource needs (5 min)

**Outputs**:

- Blockers escalated to OMNISCIENT if needed
- Scope changes logged in GitHub Projects
- Risk register updated

---

### Sprint Review & Demo (Friday)

- **Time**: 4:00 PM - 4:45 PM UTC
- **Duration**: 45 minutes
- **Day**: Every Friday
- **Platform**: Microsoft Teams (required - demo + recording)
- **Lead**: ARCHITECT + component leads
- **Attendees**: Full team + stakeholders
- **Required**: Yes for team, stakeholders (recommended)

**Agenda**:

1. Component demo (20 min)
2. Metrics review (10 min)
3. Issues closed & deliverables (10 min)
4. Q&A (5 min)

**Recording**: Saved to Teams channel for async stakeholders

---

### Sprint Retrospective (Friday)

- **Time**: 4:45 PM - 5:15 PM UTC
- **Duration**: 30 minutes
- **Day**: Every Friday (immediately after review)
- **Platform**: Microsoft Teams
- **Facilitator**: MENTOR (rotating retrospective leader)
- **Attendees**: Full team (required)
- **Required**: Yes

**Format**:

1. What went well? (5 min)
2. What could we improve? (10 min)
3. Action items for next sprint (10 min)
4. Celebration/wins (5 min)

**Outcomes**:

- Captured in GitHub Issues (retrospective tag)
- Team morale metrics
- Process improvements implemented
- Lessons learned documented

---

## 🗓️ Sprint Schedule & Key Dates

### Sprint 1.1: Distributed Inference Architecture

- **Dates**: January 6 - February 3, 2026 (4 weeks)
- **Sprint Lead**: @APEX
- **Goal**: Complete tensor parallelism, NCCL communication, orchestration framework
- **Target**: 3.8-4.2x speedup on 4 GPUs

**Key Events**:
| Date | Event | Time | Decision |
|------|-------|------|----------|
| Jan 6 | Kickoff Meeting + First Standup | 9:00-10:15 AM | - |
| Jan 8 | First Mid-Sprint | 2:00 PM | - |
| Jan 10 | First Sprint Review | 4:00 PM | - |
| Jan 17 | **Week 2 Go/No-Go Gate** | 4:00 PM | GO / NO-GO / MITIGATIONS |
| Jan 24 | Sprint 1.1 Review | 4:00 PM | - |
| Feb 3 | **Sprint 1 Completion Gate** | 4:00 PM | GO / EXTEND |

---

### Sprint 1.2: Inference Optimization & Batching

- **Dates**: February 4 - March 3, 2026 (4 weeks)
- **Sprint Lead**: @VELOCITY
- **Goal**: Batch inference, dynamic batching, quantization optimization
- **Target**: 2.5-3.0x improvement over Sprint 1.1

---

### Sprint 2.1: API & Serving Layer

- **Dates**: March 4 - April 7, 2026 (5 weeks)
- **Sprint Lead**: @SYNAPSE
- **Goal**: REST, gRPC, WebSocket APIs; request/response format standardization
- **Target**: <50ms end-to-end latency for standard requests

---

### Sprint 2.2: Monitoring & Observability

- **Dates**: April 8 - May 3, 2026 (4 weeks)
- **Sprint Lead**: @SENTRY
- **Goal**: Distributed tracing, metrics collection, alerting, SLAs
- **Target**: 99.9% availability monitoring

**Key Events**:
| Date | Event | Time | Decision |
|------|-------|------|----------|
| Feb 28 | **Mid-Phase Checkpoint (Week 8)** | 4:00 PM | Continue / Adjust |
| May 3 | **Production Readiness Gate** | 4:00 PM | GO TO MARKET / HOLD |

---

## 🌍 Time Zone Conversions

**Primary: 9:15 AM UTC**

| Region           | UTC Offset | Local Time | Alt Time           |
| ---------------- | ---------- | ---------- | ------------------ |
| UTC              | +0         | 9:15 AM    | 2:00 PM, 4:00 PM   |
| UK (GMT)         | +0         | 9:15 AM    | 2:00 PM, 4:00 PM   |
| CET (Europe)     | +1         | 10:15 AM   | 3:00 PM, 5:00 PM   |
| IST (India)      | +5:30      | 2:45 PM    | 7:30 PM, 9:30 PM   |
| SGT (Singapore)  | +8         | 5:15 PM    | 10:00 PM, 12:00 AM |
| EST (US East)    | -5         | 4:15 AM    | 9:00 AM, 11:00 AM  |
| CST (US Central) | -6         | 3:15 AM    | 8:00 AM, 10:00 AM  |
| PST (US West)    | -8         | 1:15 AM    | 6:00 AM, 8:00 AM   |

**Async Option**: For team members unable to attend live:

- Teams channel with standup thread (tagged with #standup)
- Submit answers by 12:00 PM UTC same day
- ARCHITECT reviews async submissions
- Critical blockers discussed in next sync meeting

---

## 📍 Meeting Rooms & Links

### Primary Teams Channel

- **Name**: Ryzanstein LLM-Phase3
- **Link**: [Teams Channel - Standup Discussions]
- **Recording**: All meetings recorded and archived
- **Chat**: Used for quick async updates between standups

### Recurring Meeting Schedule

#### Daily Standup

```
Meeting Link: https://teams.microsoft.com/l/meetup-join/[id]
Teams Channel: #ryzanstein-llm-phase3-standup
Recurrence: Mon-Fri, 9:15 AM UTC
Calendar Invite: YES (required)
```

#### Wednesday Mid-Sprint

```
Meeting Link: https://teams.microsoft.com/l/meetup-join/[id]
Teams Channel: #ryzanstein-llm-phase3-checkpoint
Recurrence: Every Wednesday, 2:00 PM UTC
Calendar Invite: YES (leads required, others optional)
```

#### Friday Sprint Review

```
Meeting Link: https://teams.microsoft.com/l/meetup-join/[id]
Teams Channel: #ryzanstein-llm-phase3-review
Recurrence: Every Friday, 4:00 PM UTC
Calendar Invite: YES (team required, stakeholders recommended)
Recording: Auto-saved for 90 days
```

#### Friday Retrospective

```
Meeting Link: https://teams.microsoft.com/l/meetup-join/[id]
Teams Channel: #ryzanstein-llm-phase3-retro
Recurrence: Every Friday, 4:45 PM UTC (immediately after review)
Calendar Invite: YES (required for all team)
```

---

## 👥 Facilitator Rotation

### Daily Standup Facilitators

**5-person rotation (1 week per person)**

```
Week 1 (Jan 6-10):     @APEX       - Distributed Inference
Week 2 (Jan 13-17):    @VELOCITY   - Performance Optimization
Week 3 (Jan 20-24):    @FLUX       - Infrastructure & DevOps
Week 4 (Jan 27-31):    @SYNAPSE    - API & Serving
Week 5 (Feb 3-7):      @SENTRY     - Monitoring & Observability

Then rotate: APEX → VELOCITY → FLUX → SYNAPSE → SENTRY
```

**Facilitator Responsibilities**:

- Start 2 minutes early to get technical setup ready
- Keep standup to 15 minutes strict
- Ask the 3 questions if team doesn't volunteer
- Note blockers on GitHub Issues
- Escalate critical blockers to ARCHITECT
- Capture key decisions/changes

### Wednesday Checkpoint Leaders

| Sprint | Lead      | Start Date |
| ------ | --------- | ---------- |
| 1.1    | @APEX     | Jan 8      |
| 1.2    | @VELOCITY | Feb 4      |
| 2.1    | @SYNAPSE  | Mar 4      |
| 2.2    | @SENTRY   | Apr 8      |

### Friday Sprint Review Leaders

| Week            | Primary Lead           | Support             |
| --------------- | ---------------------- | ------------------- |
| 1.1 Weeks 1-4   | @ARCHITECT + @APEX     | All component leads |
| 1.2 Weeks 5-8   | @ARCHITECT + @VELOCITY | All component leads |
| 2.1 Weeks 9-13  | @ARCHITECT + @SYNAPSE  | All component leads |
| 2.2 Weeks 14-17 | @ARCHITECT + @SENTRY   | All component leads |

### Friday Retrospective Facilitators (Rotating)

**Vary facilitator each week to bring fresh perspectives**:

```
Week 1: @MENTOR (primary)
Week 2: @APEX
Week 3: @VELOCITY
Week 4: @FLUX
Week 5: @SYNAPSE
...repeat with @SENTRY next rotation
```

---

## 📋 Meeting Preparation Checklist

### Daily Standup (5 min prep)

- [ ] Facilitator: Verify Teams link works
- [ ] Facilitator: Test audio/video 5 min early
- [ ] Team: Think about blockers before 9:15 AM
- [ ] GitHub: Update issue status if not done yet

### Mid-Sprint Checkpoint (15 min prep)

- [ ] Lead: Pull latest metrics from monitoring
- [ ] Lead: Review sprint board for scope changes
- [ ] Leads: Prepare blocker escalation list
- [ ] GitHub Projects: Ensure board up-to-date

### Sprint Review (20 min prep)

- [ ] Lead: Prepare 20-min live demo (test all components)
- [ ] Lead: Gather performance metrics from last week
- [ ] Leads: Identify completed issues to celebrate
- [ ] Leads: Prepare dashboard with sprint statistics

### Sprint Retrospective (10 min prep)

- [ ] Facilitator: Review last week's action items
- [ ] Facilitator: Send "what went well?" reflection prompt to team 24h prior
- [ ] GitHub: Create issues for agreed action items
- [ ] Facilitator: Prepare whiteboard/shared doc for voting

---

## 🚨 Escalation Triggers for Standup

**Critical** (Mention immediately in standup):

- Build failures blocking multiple team members
- Git merge conflicts affecting main branch
- Resource unavailability (GPU/infrastructure down)
- Major performance regression
- Security vulnerability
- Architectural blocker

**High** (Mention, but can wait for written follow-up):

- Design clarification needed
- Tool/setup issue preventing work
- Dependency on external team
- Timeline risk identified

**Medium** (Document in GitHub, mention if time):

- Minor bugs in development
- Code review feedback
- Documentation gaps
- Nice-to-have improvements

---

## ✅ Meeting Success Criteria

### Daily Standup

- ✓ Started on time (<2 min late)
- ✓ Finished in ≤15 minutes
- ✓ All blockers identified
- ✓ All blockers logged in GitHub Issues
- ✓ Attendance ≥90%

### Mid-Sprint Checkpoint

- ✓ Finished in ≤30 minutes
- ✓ Metrics dashboard reviewed
- ✓ Scope changes documented
- ✓ Escalations assigned owners & SLAs
- ✓ Key decisions captured in GitHub

### Sprint Review

- ✓ Demo completed successfully
- ✓ Metrics reviewed against targets
- ✓ All completed work celebrated
- ✓ Recording uploaded within 1 hour
- ✓ Stakeholder feedback captured

### Sprint Retrospective

- ✓ Finished in ≤30 minutes
- ✓ At least 3 action items identified
- ✓ Action items assigned owners
- ✓ Team morale assessed
- ✓ Lessons learned documented

---

## 📊 Metrics Tracked in Sprint Reviews

**Performance Metrics**:

- GPU utilization (target: >85%)
- Speedup vs baseline (target: 3.8-4.2x for 1.1)
- Latency p50/p95/p99 (tracked per sprint)
- Memory efficiency (% of available)

**Development Metrics**:

- Issues closed (vs sprint commitment)
- Code coverage (target: >90%)
- Test pass rate (target: 100%)
- Code review turn-around (target: <24 hours)

**Team Metrics**:

- Standup attendance (target: >95%)
- Blocker resolution time (target: <24 hours)
- Regressions introduced (target: 0)
- Documentation currency (target: ≤2 days behind code)

---

## 🔄 Standup Archive & Knowledge Base

**GitHub Issues Labels for Standups**:

- `standup` - Daily standup notes
- `blocker` - Current blockers
- `checkpoint` - Mid-sprint checkpoint notes
- `retrospective` - Retro insights & action items
- `decision` - Key decisions from standups
- `escalation` - Escalated issues

**Standup Notes Format** (captured by facilitator):

```markdown
## Standup - [Date]

### Completed

- [Name]: [Work completed]
- [Name]: [Work completed]

### In Progress

- [Name]: [Work planned]
- [Name]: [Work planned]

### Blockers

- [Name]: [Blocker] - @[Owner] - [ETA]
- [Name]: [Blocker] - @[Owner] - [ETA]

### Decisions Made

- [Decision and owner]

### Next Actions

- [Action] - @[Owner] - [Due date]
```

---

## 📞 Out-of-Band Escalation

If a **critical blocker** arises between standups:

1. **Notify Immediately** (in Teams channel):

   - Tag @ARCHITECT + @OMNISCIENT
   - Mark with 🚨 emoji
   - Provide 2-sentence summary

2. **Response SLA**:

   - Critical: Response in <30 minutes
   - Escalation path: ARCHITECT → @OMNISCIENT → specialist agent

3. **Resolution**:
   - Synchronous call scheduled within 2 hours if needed
   - Decision documented in decision log
   - Team notified in next standup

---

**Phase 3 standup cadence established. All meetings scheduled with facilitation, escalation paths, and success criteria defined. Ready for January 6 kickoff.**
