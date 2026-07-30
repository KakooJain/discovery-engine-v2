TAXONOMY_PROMPT = """---
Tag Indian quick-commerce reviews. Return ONLY JSON with keys: behavioral_driver, discovery_barrier, discovery_channel, frustration_type, segment_marker, unmet_need, categories, sentiment_score, supporting_quote.
CRITICAL: assign tags only on direct explicit evidence; do not infer. Empty arrays are expected when uncertain.

behavioral_driver tags: habit-routine, trust-familiarity, convenience-speed, price-sensitivity | cues: same items, reorder, known brands, hurry, no time, cheap, sasta, budget, discount
discovery_barrier tags: discovery-friction, trust-deficit-new-category, info-gap, risk-aversion, no-need-perceived, habit-lock-in | cues: hard find, poor search, authenticity doubt, missing details, afraid try, not needed, repetitive recommendations, only order the same few things, never see anything else, same products every time, hard to browse, don't know what else they have, app only shows what I usually buy, same items every time, app shows me the same things, nothing new ever appears, reorder is the only thing I use, homepage never changes, only use it for X, just for groceries, wouldn't buy Y from a grocery app, how do I know if it's good, no way to compare, can't tell the quality, not sure which one to pick
discovery_channel tags: algo-discovery, social-discovery, search-driven, offline-to-online, word-of-mouth | cues: for-you, instagram, influencer, youtube, search bar, search kiya, saw offline, friend referral
frustration_type tags: stockout-availability, quality-freshness, delivery-experience, pricing-surge, app-ux-bug, cs-support, return-refund | cues: out of stock, stale, late, overpriced, crash, no response, refund delay
segment_marker tags: power-user, occasional-user, new-user, price-conscious, premium-experimenter, parent-household, pet-owner, single-professional | cues: daily, rarely, first order, budget-focused, gourmet/imported, baby/kids, dog/cat, bachelor
unmet_need tags: unmet-need-category, unmet-need-bundling, unmet-need-trust-signal | cues: please add, missing category, combo/bundle, trial size, certification, verified proof, wish they sold, should also have, would be nice if they added, why no X
categories: groceries, fruits-vegetables, dairy, snacks-beverages, household-essentials, personal-care, baby-care, pet-supplies, electronics, home-decor, stationery, pharmacy, meat-seafood, bakery, frozen-food

sentiment_score: -2 furious,bakwas,scam,fraud,never again; -1 dissatisfied complaint; 0 neutral/mixed; +1 satisfied good works fine; +2 delighted best love amazing.
supporting_quote: one verbatim diagnostic fragment, 5-15 words (max 20), never full review.
---"""