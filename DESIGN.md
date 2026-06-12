# The Design of Reward2Reason

*A document for course builders: what this course is optimized for, the
mechanisms it uses, and the decisions behind them — including the places
where honest engineering forced the narrative to change.*

## The thesis

Nand2Tetris is not a great course because of its content. Computer
architecture was taught for decades before it. It is great because it is
an **engineered sequence of revelations**: the student repeatedly
discovers that something apparently impossible is buildable *by them,
today*, from parts they already understand. The feeling it produces —
"I shouldn't be able to do this, but somehow I can" — is the product.
Knowledge is the exhaust.

Reward2Reason bets that the same architecture works for machine
intelligence, with:

- **primitive:** a scalar reward signal *(the NAND gate)*
- **summit:** an engine that masters games through self-play
  *(the Tetris)*
- **emergence story:** reward → preference → behaviour → foresight →
  intuition → imagination → reason

## The ten mechanisms

1. **One primitive, total radical compression.** Everything in the
   course is the nudge rule `x += lr * (target - x)` wearing
   progressively better disguises — and the course *says so*, in Module
   1, and then proves it ten times. The student can hold the whole tower
   in one hand. That graspability is where "computers aren't magic
   anymore" comes from in Nand2Tetris, and where "intelligence isn't
   magic anymore" comes from here.

2. **The boss fight inversion (Module 0).** Nand2Tetris puts the summit
   in the title. We go further: the student *plays against the finished
   artifact* in the first ten minutes, and loses. This converts the
   course's promise from an abstraction into a grudge. The final module
   closes the loop with a rematch against a machine the student built
   and trained. Long-range motivation with a personal score to settle.

3. **Walls before doors.** The single most copied-out-of-order idea in
   course design is "motivate the technique." We enforce a stronger
   form: the student must *personally hit and measure* each limitation
   before the fix is even named. Module 5 is the purest case — an entire
   module that builds nothing but measuring instruments and uses them to
   watch the student's own Module 4 agent fail a fair exam. The fix
   (Module 7) then lands as a rescue, not a topic.

4. **Invent first, name after.** Bellman, TD-error, DQN, PUCT, AlphaZero
   — every formal name in this course appears *after* the student has
   built the thing, in a section literally called "The reveal." The
   emotional payload of "this thing you wrote on Tuesday is what
   dopamine neurons compute / is the 2017 Nature paper" only fires in
   that order. Terminology-first teaching spends that payload for
   nothing.

5. **The Built-In Chip Rule.** Stolen from Nand2Tetris and applied with
   teeth: hand-built first, then the fast provided version is *earned*
   (Dial → `r2r.tensor`; your MCTS → the built-in boss). This converts
   "using a library" from a defeat into a graduation, and teaches the
   profession's deepest habit — trusting abstractions because you've
   been below them, not because you have to.

6. **Tests as examiner, demos as theater.** Each module has two
   distinct verifications with distinct emotional jobs. `pytest` is the
   chip-tester: cold, binary, trustworthy (in Module 6 the tests check
   the student's gradients against finite differences — the examiner is
   calculus). The `__main__` demo is the victory lap: it *shows* the
   ALU computing — the map condensing, the cliff edge walked, the
   perfect player held to 100 draws — and ends by pointing at the next
   wall. Green light, then goosebumps, then cliffhanger.

7. **Every module ends running.** No dead zones, no "this will be
   useful later." The dependency chain is executable: Module 4's demo
   *runs the student's Module 3 agent* as its baseline; Module 7's exam
   uses the instruments built in Module 5; Module 10 is assembled from
   Modules 7, 8, and 9 by name.

8. **The opponent ladder is the curriculum (Act III).** Random → your
   past self → blind MCTS → perfect minimax → the boss → you. From
   Module 8 onward, progress is measured in an arena with an Elo
   ladder, and the deepest rung is the Hall of Mirrors: the student's
   own checkpoints, ranked, so they can watch yesterday's self get
   dismantled by today's. Self-play is also the course's own meta-joke:
   the curriculum that tracks you perfectly is the thing being taught.

9. **Identity, stated as fact, at the end.** Module 11 contains no
   code. It replays the eleven moves, names the one empirical fact the
   student has now *personally verified* (optimization through
   composed layers buys understanding-shaped behaviour), and then
   delivers the modern punchline: the assistant they consulted during
   the course was trained by Module 1's primitive at planetary scale
   (RLHF), and the frontier — reasoning models — is Module 10's
   marriage performed on language. The student doesn't leave knowing
   RL. They leave *being someone who has built a mind-shaped thing*,
   standing on a map where "you are here" points at the frontier.

10. **Scale chosen for the desk lamp.** Every claim in the course is
    verified on a laptop CPU in seconds-to-minutes (the one indulgence,
    the Connect Four boss run, is ~an hour and is framed as a campfire,
    not a prerequisite). This is a hard design constraint, not an
    accident: awe requires *witnessing*, and witnessing requires the
    loop to close while the student is still watching.

## Decisions, including the ones reality forced

**Connect Four, not chess.** The capstone *game* is fungible; the
capstone *loop* is not. Chess from scratch in pure numpy trains for
weeks and the student watches none of it. Connect Four's engine trains
visibly in an evening, beats its creator, and the `Game` interface makes
"now try a chess variant" the natural epilogue exercise. Choose the
summit by what can be *witnessed*, not by prestige.

**Tabular Monte Carlo before TD, and an honest Module 4.** The first
design had Module 4 demonstrate "TD learns faster than Monte Carlo" on
a big maze. When implemented and measured, *the claim was false* — in
deterministic sparse-reward mazes, episode-level credit assignment is
genuinely competitive. The module was rebuilt around the claim that is
true and deeper: off-policy TD learns *what the world is* rather than
*what happened to you*, demonstrated on the cliff walk, where the two
agents walk visibly different routes for visibly different reasons. The
general law for course builders: **run every demo before you write its
lesson; when the demo disagrees with the textbook story, the demo is
the lesson.**

**A 16×16 moving-goal room as The Wall.** The wall world must be (a)
honestly fatal to tables — not "slow," *fatal at any humane budget*;
(b) trivially soluble by a tiny MLP, so the rescue is immediate and
total (16% → 100% on fresh exams, with *less* experience); (c) small
enough to train in a minute. Measured calibration, not vibes: the 12×12
version let the table limp to 54%, which is a muddle, not a wall.

**Scalar autograd, then vectorized.** Module 6 builds micrograd-style
scalar backprop (the idea, naked), and the Built-In Chip Rule then
grants the numpy tensor version (the same idea, fast). Building the
tensor version by hand teaches broadcasting headaches, not insight;
building neither teaches dependence.

**Both search paradigms, as a duel.** Minimax (exhaustive, dies of
combinatorics — Module 5's wall in a new mask) and MCTS (budgeted,
scales smoothly) are built side by side, because the capstone's PUCT is
only legible as "your MCTS with intuition installed." Bonus rhyme: UCB
inside the tree is Module 2's casino again. Courses this shape should
rhyme — rhymes are compression, and compression is the thesis.

**The student never trains against a teacher.** At no point does the
agent learn from expert games, and the course says so loudly in Modules
0, 8, and 10 — because "nobody taught it" is the single load-bearing
fact under the final awe. Protect that fact from convenience features
at any cost.

**Solutions ship in-repo.** A course this test-driven needs its own CI
(`R2R_SOLUTIONS=1 pytest`), and pretending students can't find solutions
on the internet is naive. The defense is framing (the README is blunt
about what peeking costs) plus design: the artifact of each module is
the *experience of the wall breaking*, which cannot be copy-pasted.

## The reusable template

To build "X2Y" for another field:

1. Pick the summit **Y**: an artifact a beginner would call impossible,
   demonstrable in under a minute, and *personal* — the student should
   be able to lose to it, be moved by it, or show it to a friend.
2. Find the primitive **X**: the smallest piece such that *every* layer
   of Y is X composed with itself. If you need a second primitive,
   you haven't found X yet.
3. Walk backward from Y to X; at each step ask, "what is the smallest
   capability whose absence makes the next layer *measurably*
   impossible?" Those measurements are your wall-modules. Build the
   instruments into the course.
4. Write every demo first. Run it. When it disagrees with the story you
   planned, change the story — the demo that's true is always more
   interesting than the claim that isn't.
5. Give every module a green light (tests), a spectacle (demo), a
   reveal (real name + lineage), and a cliffhanger (the next wall,
   already cracked open in this module's last paragraph).
6. End with identity, not assessment: a module whose only content is
   the view — what the student has become, where they now stand, and
   the one step that's still unclimbed, pointed at like an invitation.

The formula in one line:

> **Don't teach the field. Stage the emergence, and make the student
> the one who commits every miracle.**
