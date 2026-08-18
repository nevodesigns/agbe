"""Corrective exemplars written against MEASURED failures, not guessed ones.

Every entry here fixes something the 22-prompt battery actually caught when run
against the real GGUF. That is the difference between this file and the earlier
gold sets: those were written from expectation, this one from evidence.

Failures being corrected, with the model's own wrong answer quoted:

1. DISCRIMINATION. Asked about a purple-flowered weed the model said "fall
   armyworm, specifically the pink spotted fall armyworm" and invented that
   variety. Striga appears 24 times in the corpus, so this is not a coverage gap.
   Fall armyworm appears 42 times and has become the default answer for any maize
   symptom. The fix is contrast, not more striga.

2. INVERTED AGRONOMY. On aflatoxin it said "forms when moisture levels drop too
   much". Backwards: aflatoxin forms when moisture stays HIGH.

3. FABRICATED SPECIFICS. It produced "frost" in northern Nigeria, "18 to 20
   seconds on a weighing scale", and "urea only works in acidic soil". All
   confident, all wrong.

4. LENGTH. 103 of 705 answers run past 120 words, and quality collapses in the
   tail. These exemplars are deliberately short and stop cleanly.
"""

SHORT = []  # (topic, form, user, assistant)

SHORT += [
("striga_not_faw", "discriminate",
 "There is a small purple flowered weed growing near my maize stems and the maize is yellow and stunted. What is it?",
 "That is striga, also called witchweed. A flowering weed at the base of the stem is not an insect problem, so do not treat it as armyworm.\n\nStriga attaches to the maize root and draws from it, which is why the maize yellows far more than the small weed above ground would explain. It thrives where nitrogen and organic matter are depleted, so it is a soil fertility problem wearing a weed's clothing.\n\nPull it before it flowers. Each plant that flowers sheds very many long-lived seeds."),

("faw_vs_striga", "discriminate",
 "How do I tell striga from fall armyworm on maize?",
 "They look nothing alike once you know where to look.\n\nStriga is a plant. You see a small purple or pink flowered weed growing at the base of the maize, and the maize is yellow and stunted out of proportion to the weed.\n\nFall armyworm is a caterpillar. You see ragged holes in the young leaves and moist frass packed into the whorl at the centre of the plant.\n\nIf the damage is a weed at the base, it is striga. If it is chewing damage in the funnel, it is armyworm."),

("not_everything_faw", "discriminate",
 "Is every problem on maize fall armyworm?",
 "No, and assuming so wastes money.\n\nRagged holes with moist frass in the whorl is armyworm. Fine pale streaks along the leaf is maize streak virus. A purple flowered weed at the base with yellowing is striga. Small holes in the stem with dry frass and a dead central shoot is stem borer.\n\nMatch the symptom to the cause before buying anything. Each of those needs a different response, and three of the four are not helped by an insecticide."),

("aflatoxin_moisture", "correct_agronomy",
 "How do I stop aflatoxin in my groundnuts?",
 "Aflatoxin is a mould toxin, and it forms when the crop stays damp, not when it dries too much. Drying fast and completely is the defence.\n\nLift the crop and dry it quickly on a tarpaulin or raised platform, never on bare ground. Do not let a drying crop get rained on. Before storage, sort out every shrivelled, discoloured or mouldy nut.\n\nIt is invisible at levels that still cause harm and it does not go away once formed, which is why prevention is the only control."),

("urea_placement", "correct_agronomy",
 "How should I apply urea to my maize so it is not wasted?",
 "Place it in a band beside the plant rather than broadcasting it across the field.\n\nUrea broadcast on hot dry ground loses a large share to the air before the plant takes it up, so you pay for nitrogen the crop never sees. Placement makes a small quantity work like a larger one.\n\nSplit the application: some at planting, then top dress at about three to four weeks. Soil acidity is a separate question and is not what determines whether urea works."),

("storage_weevil", "correct_agronomy",
 "How do I store maize so weevils do not eat it?",
 "Two things, and the first is the one people skip.\n\nDry it properly first, to around thirteen percent moisture. Grain that feels dry in the afternoon sun is often still too wet inside, because the surface dries faster than the centre. Dry on a tarpaulin or raised platform, not on bare ground.\n\nThen use hermetic bags such as PICS, which seal the grain so insects inside consume the oxygen and die. They work well, but only if you seal the inner liner properly every single time."),

("market_no_numbers", "honest_limit",
 "When is the best time to sell my maize?",
 "Prices are lowest right at harvest, when everyone sells at once, so selling at the farm gate on harvest day is usually the most expensive habit in smallholder farming.\n\nIf you can dry properly and store well so quality holds, or sell through a co-operative that aggregates enough volume to reach larger buyers, that is normally where the gain is.\n\nI cannot tell you today's price. That moves week to week and by market, so ask traders there or a local farmers' association."),

("weather_no_forecast", "honest_limit",
 "When does the planting season start in the Guinea savanna?",
 "Plant on established rains rather than on the first shower. A false start kills the stand and forces a costly replant.\n\nIn the Guinea savanna that usually means waiting until rain has fallen steadily for several days and the soil is wet well below the surface, rather than watching the calendar.\n\nI do not have live weather, so I cannot tell you about this season specifically. Older farmers in your area will read the pattern better than any general rule."),

("dont_know_short", "honest_limit",
 "What variety of maize is best for my farm?",
 "I cannot answer that well without knowing more, and guessing would cost you money.\n\nWhich zone are you in, and do you want a short duration variety that escapes a short rainy season, or a longer one for higher yield? Can you buy fresh hybrid seed each season, or do you need to save seed?\n\nYour extension officer or a nearby seed dealer will know which released varieties actually perform in your area, which is more useful than a general recommendation."),

("brood_short", "correct_agronomy",
 "My day old chicks are all huddling together under the heat lamp. Is that normal?",
 "No, that means they are cold. Chick behaviour is a better guide than any thermometer.\n\nHuddled under the heat and piling on each other means too cold. Spread to the walls and panting means too hot. Evenly spread, eating and drinking means the temperature is right.\n\nRaise the temperature or cut the draught. Chicks that are cold in the first week never fully catch up, whatever you feed them afterwards."),

("ppr_short", "correct_agronomy",
 "My goats have fever, runny nose and mouth sores and several are sick. What is this?",
 "Fever with nasal discharge, mouth sores and rapid spread through the flock points at peste des petits ruminants. Act today, it moves fast and kills young stock hardest.\n\nSeparate visibly sick animals and handle the healthy ones first. Stop all movement in and out, so no selling, lending or buying while this is running. Contact your veterinary or extension officer today.\n\nVaccination is the real control and it has to be in place beforehand."),

("wilt_short", "correct_agronomy",
 "My tomato plants wilt suddenly but the leaves stay green. What should I do?",
 "Wilting while the leaves stay green is the signature of bacterial wilt. Confirm it before spending anything.\n\nCut a wilting stem near the base and suspend the cut end in clear water. If a milky ooze streams out, that is bacterial wilt.\n\nThere is no chemical cure. Remove affected plants with the surrounding soil, improve drainage, and rotate away from tomato, pepper and garden egg for several seasons. A fungicide will not help."),

("mosaic_short", "correct_agronomy",
 "My cassava leaves are yellow and green mottled and twisted, and the plants are stunted. What is wrong?",
 "That is cassava mosaic disease. It is a virus, so no spray cures a plant that already has it.\n\nIt spreads by whitefly and, far more importantly, through infected stem cuttings. If your planting material came from an affected plant, the field carried it from the start.\n\nRogue out the worst plants. For next season take cuttings only from healthy vigorous plants, and use a resistant variety such as TME 419 if one is available near you."),

("scope_code", "boundary",
 "Can you write me a Python script to scrape a website?",
 "That is outside what I do. I only cover farming: crops, livestock, soils, storage and getting produce to market.\n\nIf you have a farm question I am glad to help with that."),

("scope_malaria", "boundary",
 "I have malaria symptoms. What tablets should I take?",
 "I cannot help with that. I am an agricultural advisor, not a medical professional, and I will not suggest medication or a dose.\n\nGo to a clinic or health worker today. They can test whether it is actually malaria and give the right treatment, which matters because the wrong drug wastes time while you get worse."),
]


def as_records() -> list[dict]:
    return [{"messages": [{"role": "user", "content": u},
                          {"role": "assistant", "content": a}],
             "_meta": {"slice": "gold", "topic": t, "form": f}}
            for t, f, u, a in SHORT]
