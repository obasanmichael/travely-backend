# How the Travely recommendation engine works

Engine version **3.0.0** · code in `engine/` · tests in `tests/test_engine_*.py`

Travely answers one question: *given where you're starting from, when, for how
long, with whom, how much you can spend and what you enjoy, which places in
Nigeria should you go to?* The engine answers it in four stages, using real,
sourced data only. Every number it shows can be traced back to where it came
from, and nothing missing is ever filled in with a guess.

```
 quiz answers ──▶ 1. hard constraints ──▶ 2. fuzzy practicality ──▶ 3. interest match ──▶ 4. rank, diversify, explain
                    (can this work?)        (how practical is it?)    (will they enjoy it?)   (top picks + reasons)
```

---

## Inputs

| Answer | Used for |
|---|---|
| Starting state | Distance, travel time and the intercity fare |
| Travel month, or "flexible" | Season fit |
| Number of days | Nights to pay for, food and transport days, travel burden |
| Number of travellers | Every per-person cost, and rooms (2 people per room) |
| Total budget (₦, for the group) | Affordability. There is **no cap**: the engine works in ratios. |
| Comfort level: budget / mid / premium | Accommodation tier and eating-out markup |
| Interests, each marked *love* or *like* | The interest profile |
| Pace: relaxed / balanced / packed | Nudges the profile towards relaxation or adventure |
| Travelling with children | Boosts family-friendly places |
| Longest acceptable one-way travel time (optional) | A hard limit |

## The catalog

Destinations come from Wikidata (CC0). Each one is reviewed by a person before it is
**published**, and only published places are recommended. Each destination
carries:

- **weighted tags** from a fixed 28-tag taxonomy (`engine/taxonomy.py`), e.g.
  Obudu = nature 1.0, hiking 0.8, waterfalls 0.4. The quiz offers the *same*
  tags, so users and places are compared like for like.
- **12 months of climate normals** (NASA POWER): temperature and rainfall.
- **a credited photo** (Wikimedia Commons, commercial-reuse licences only).
- a **quality prior**: Wikipedia article (0.5) + photo (0.3) + description
  (0.2). It's only used to separate otherwise similar options.

---

## The cost model (`engine/costs.py`)

Every trip cost is a **range** (low–high) with a **confidence** level
(high / medium / low) and a plain-language **basis** for each part.

| Component | How it's estimated | Source |
|---|---|---|
| Getting there and back | The origin state's average intercity bus fare, scaled by road distance, return, per traveller. Zero within the same state. | NBS Transport Fare Watch (monthly, by state) |
| Getting around | In-city bus fare × 2–6 trips a day × days × travellers | NBS Transport Fare Watch |
| Food | A plate of rice and stew with beef, priced from zonal ingredient prices, × an eating-out markup for the comfort tier, × 3 meals × days × travellers | NBS Selected Food Price Watch (monthly, by zone) |
| Accommodation | p25–p75 of verified nightly rates for the tier at the destination, falling back to city then state, × nights × rooms. Zero for day trips. | Curated observations, each with a source and date |
| Entry fees | Verified fees per person. Shown as "not yet verified" when unknown. | Curated observations |

### Assumptions (stated so they can be challenged)

| Assumption | Value | Why |
|---|---|---|
| Road distance vs straight line | × 1.3 | A typical detour factor for road networks |
| Distance the NBS intercity average represents | ~300 km, scale clamped to 0.5–3× | NBS gives one average per state, not per route |
| Average road speed | 55 km/h | Nigerian intercity roads, including stops |
| In-city trips a day | 2–6 | Range from a quiet day to a busy one |
| Eating-out markup on ingredients | budget 1.6–2.4×, mid 3–5×, premium 6–10× | Buka vs restaurant vs hotel dining. To be calibrated with observed meal prices. |
| Freshness window | 180 days for NBS series (they publish months behind), 120 days for our own observations | Older data lowers confidence |

### Confidence
- **Missing component** (usually accommodation): the estimate is incomplete
  and the destination is **not recommended**. We don't guess.
- **Modelled components** (food, distance-scaled transport) cap confidence at **medium**.
- **Stale data**, thin evidence (fewer than 3 rates) or state-level rates lower confidence further.

### What we don't do
- **LiteAPI hotel rates are never fed into the cost model.** Their terms allow
  live display and booking only. They appear as a separate live "Places to
  stay" panel, not in our estimates.

---

## Stage 1: hard constraints (prevent, don't just detect)

A place is removed if:
1. its cost is incomplete (no data, so no recommendation), or
2. its **cheapest** estimate is more than **115%** of the budget, or
3. it's beyond the user's travel-time limit.

If fewer places remain than the user asked for, constraints are relaxed **in a
fixed order**: the budget tolerance to 135%, then the travel-time limit. **The
response says exactly what was relaxed**, and each affected pick says it's
over budget. Cost-unknown places are never let back in.

If nothing fits, the response says so and gives the **cheapest feasible budget**,
never an empty screen.

## Stage 2: fuzzy practicality (`engine/fuzzy.py`)

"Affordable", "good season" and "too far for a weekend" are vague human ideas
with no sharp cut-off. A ₦39,900 vs ₦40,100 budget shouldn't flip a
recommendation. Fuzzy logic models that gradient, and its rule table can be
read by someone who doesn't write code.

**Inputs** (all ratios, so no ₦ cap):
- **affordability** = budget ÷ estimated cost (mid-point) → *tight / comfortable / generous*
- **season fit** (0–1) → *poor / fair / ideal*
- **travel burden** = share of the trip's waking hours spent on the road → *light / moderate / heavy*

**Output:** practicality 0–1 → *low / medium / high*

**Rules** (the whole policy):

| If | Then practicality is |
|---|---|
| generous AND ideal season AND light travel | high |
| comfortable AND ideal season AND light travel | high |
| generous AND fair season AND light travel | high |
| comfortable AND fair season AND NOT heavy travel | medium |
| (comfortable OR generous) AND ideal season AND moderate travel | medium |
| generous AND fair season AND moderate travel | medium |
| tight AND ideal season AND light travel | medium |
| tight AND (NOT ideal season OR NOT light travel) | low |
| poor season | low |
| heavy travel AND NOT generous | low |
| heavy travel AND generous AND ideal season | medium |
| heavy travel AND generous AND NOT ideal season | low |

It's a standard Mamdani system (min for AND, max for OR, 1−x for NOT, max
aggregation, centroid). It's evaluated with numpy across all candidates in one
pass, and tests check it against scikit-fuzzy's own engine (within 0.001).

**Season fit** (`engine/season.py`) comes from the climate data. Rain at or below
50 mm a month is ideal and 250 mm+ is worst; heat above 30°C costs a little. It
is weighted by how *outdoor* a place is, so a museum barely cares about rain and
a waterfall trail does. "Flexible" uses the place's best month.

**Budget band** (Low/Medium/High per person per day) comes from one fuzzy
definition served by the API. *Provisional until accommodation rates are in.*

## Stage 3: interest match (`engine/content.py`)

The user becomes a vector over the same taxonomy: *love* = 1.0, *like* = 0.6.
Pace adds relaxation (relaxed) or adventure/hiking (packed). Children add
family-friendly. **Cosine similarity** with each destination's tag vector is
the content score.

## Stage 4: rank, diversify, explain (`engine/ranker.py`)

`score = 0.55 × content + 0.35 × practicality + 0.10 × quality`

With no interests given, the weights become 0 / 0.75 / 0.25 and results are
labelled "popular picks".

**Diversity: Maximal Marginal Relevance** (λ = 0.6). After the best match, each
next pick trades a little relevance for being different from what's already
picked (by tags, and by being in the same state). Variety is only chosen
**among places that match the user's interests**. Unrelated places only fill
slots nothing relevant could.

**Reasons** are built from each pick's own numbers, never generated text:
- "Matches your interest in waterfalls, hiking"
- "About ₦120,000 under your budget"
- "January is a great time to visit" / "Better in December; August is rainy here"
- "About 6 hrs by road from Lagos"
- "Cost estimate confidence: medium"

---

## Guarantees the tests enforce

| Property | Test |
|---|---|
| Unpriced places are never recommended | `test_unpriced_destinations_are_never_recommended` |
| Results stay within budget tolerance; relaxation is disclosed | `test_results_stay_within_budget_tolerance`, `test_relaxation_is_disclosed` |
| More budget never removes an affordable result | `test_raising_budget_never_removes_an_affordable_result` |
| More money never lowers practicality | `test_more_money_never_makes_a_trip_less_practical` |
| Small budget changes don't flip scores | `test_small_budget_changes_dont_flip_the_score` |
| Every input combination fires a fuzzy rule | `test_every_input_combination_fires_a_rule` |
| Rainy months lower outdoor practicality | `test_rainy_month_lowers_outdoor_practicality` |
| No three near-identical picks | `test_diversity_avoids_three_near_identical_waterfalls` |
| Variety never beats relevance | `test_variety_never_beats_relevance` |
| Nothing affordable → guidance, not an empty list | `test_nothing_affordable_returns_guidance_not_an_empty_shrug` |

## Known limitations
- Food cost is modelled from ingredient prices, not observed meal prices.
- NASA POWER's ~50 km grid reads highland spots (Obudu, Mambilla) warmer than they are.
- Travel time assumes road travel; flights aren't offered as an option yet.
- The weights (0.55 / 0.35 / 0.10) and λ are reasoned choices, not yet tuned
  on real user behaviour. Saves and dismissals are logged for that.
