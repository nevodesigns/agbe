"""Nigerian Pidgin exemplars and fall armyworm identification, hand-written.

v3 passed the refusal and the timing prompt but failed two:

  1. **A Pidgin question got an English answer.** Pidgin was 3% of the corpus
     (18 of 601). The refusal only started working once refusals reached 7% and
     were oversampled, so language selection needs the same weight. Every pair
     here is question AND answer in Pidgin, so the model learns to match the
     language it was asked in rather than merely to recognise Pidgin.

  2. **"Pod borers" instead of fall armyworm.** The model had "wet sawdust is
     frass" from our corpus but attached it to the wrong pest, because fall
     armyworm appeared in only about ten examples and competed with every other
     boring insect. These pairs name it explicitly across many phrasings, and
     several contrast it with the pests it is genuinely confused with.

Hand-written because templating cannot produce idiomatic Pidgin, and because
distinguishing two similar pests requires stating the actual difference.
"""

from __future__ import annotations

# (topic, form, [(user, assistant), ...])
PAIRS: list[tuple[str, str, list[tuple[str, str]]]] = [

    # ---------------- Pidgin: pests -------------------------------------
    ("pidgin_faw_2", "pidgin", [
        ("Wetin dey chop my maize leaf? I see white worm for inside the middle.",
         "Na fall armyworm. Na the caterpillar dey stay for inside the whorl, the middle of the plant, and na there e dey chop.\n\nHow you go know say na am:\n\n1. Open the middle of some plant for morning or evening time, when dem dey active.\n2. The caterpillar head get white mark like Y wey turn upside down.\n3. Near the tail, you go see four black dot wey stand like square.\n\nIf the farm small, pick dem comot with hand for some days. E dey work well. After harvest, clear and burn the stalk, na there dem dey hide reach next season.\n\nIf you go spray, use product wey dem register for fall armyworm for maize, follow the rate wey dey for label, and pour am enter the middle. Spray wey you pour on top leaf no go reach the caterpillar."),
    ]),

    ("pidgin_striga", "pidgin", [
        ("Small purple flower dey grow near my maize and the maize don yellow and no dey grow well.",
         "Na striga, wetin some people dey call witchweed. E dey attach to the root of the maize and dey suck from am, na why the maize yellow pass wetin the small weed suppose cause.\n\nThe main thing wey you suppose sabi: striga dey grow well for land wey don tire, wey nitrogen and organic matter don finish. So na soil problem wear weed cloth.\n\nWetin you go do:\n\n1. Pull am comot before e flower. Every one wey flower dey drop plenty seed wey fit stay for ground for years.\n2. Rotate with cowpea or soybean, and put organic matter back for the soil. As the soil dey better, striga dey reduce.\n3. If resistant variety dey your area, use am, but e no go work alone if the soil still tire."),
    ]),

    ("pidgin_tomato", "pidgin", [
        ("My tomato dey wilt but the leaf still green. Wetin I go spray?",
         "Before you buy anything, make you test am first, because if na wetin I dey think, no spray go help and you go just waste money.\n\nCut the stem wey dey wilt near the base, put the cut side inside clear water for glass. Watch am for one or two minute. If you see white thing like milk dey comot enter the water, na bacterial wilt.\n\nWilt wey the leaf still green na the sign. No chemical dey cure am. Wetin you fit do:\n\n1. Comot the plant with the soil around the root, and no put am for compost.\n2. No plant tomato, pepper or garden egg for that ground for some seasons.\n3. Make the drainage better, because the bacteria dey move for wet soil.\n\nIf the milk thing no comot, then na something else. Tell me wetin you see and we go check am."),
    ]),

    ("pidgin_storage_2", "pidgin", [
        ("How I go know say my maize don dry reach to store?",
         "Maize wey feel dry for afternoon sun fit still wet too much for inside. Na the outside dey dry first, the inside dey hold water.\n\nWetin you fit do to check:\n\n1. Bite the grain. If e hard well well and e crack sharp, e don near dry. If e soft or e dey bend, e never ready.\n2. Pour some inside empty tin. Dry grain dey make sharp sound; wet one dey sound dull.\n3. If co-operative near you get moisture meter, na the correct way to check.\n\nMake you dry am for tarpaulin or raised platform, no be for bare ground. Ground dey push water back enter the grain and dey add sand wey buyer no go like."),
    ]),

    ("pidgin_poultry_2", "pidgin", [
        ("My broiler chick dem dey gather for one place under the light. Wetin dey happen?",
         "Dem dey cold. The way chick dey arrange demself na the best thermometer wey you get.\n\n1. If dem gather under the heat and dey pile on top each other, the place too cold. Bring the heat down small or block draught.\n2. If dem run go the wall and dey open mouth dey breathe, e too hot. Raise the heat up.\n3. If dem spread evenly and dey chop and drink, na correct temperature be that.\n\nStart warm for first week, then reduce am small small every week as dem dey grow feather. Chick wey cold for first week no dey ever catch up well, no matter the feed wey you give am after."),
    ]),

    ("pidgin_cost", "pidgin", [
        ("I no get money this season at all. Wetin I fit still do for my maize?",
         "Weed am on time. If na only one thing you fit do, na that one.\n\nWeeding no dey cost money, na your labour. And for small maize farm, e dey change the yield pass any input wey you fit buy with the money wey you no get. Maize dey suffer most from weed for the first few weeks, so early weeding matter pass plenty weeding later.\n\nAfter that, the things wey no cost money:\n\n1. Correct spacing, one plant for one hole, 75cm between row and 25cm between plant.\n2. Plant when rain don settle, no be first shower. That one go save you from planting twice.\n3. If small money dey, put small urea for side of the plant for three to four weeks, no be broadcast. Placement dey make small quantity work like plenty."),
    ]),

    # ------------- fall armyworm, named explicitly, many phrasings --------
    ("faw_named_1", "diagnose", [
        ("There are ragged holes in my young maize and something wet and grainy sitting in the funnel of the plant.",
         "That is fall armyworm. The wet grainy material in the funnel is frass, the caterpillar's droppings, and finding it packed into the whorl is the identifying sign.\n\nConfirm it before treating:\n\n- Open the whorl on several plants across the field, early morning or late evening when the larvae are active.\n- The larva has a pale upside-down Y marking on the head capsule.\n- There are four dark spots arranged in a square near the tail end.\n\nCount affected plants out of twenty. That proportion, not the presence of one caterpillar, is what decides whether treatment is worth the cost."),
    ]),

    ("faw_vs_stemborer", "compare", [
        ("How do I tell fall armyworm apart from stem borer?",
         "They damage different parts of the plant, and that is the reliable way to separate them.\n\n**Fall armyworm** feeds down inside the whorl, the funnel at the growing point. You see ragged holes across the young leaves as they unfurl, and moist frass packed in the centre. The larva has a pale inverted Y on the head and four dark spots in a square near the tail.\n\n**Stem borer** tunnels into the stem itself. You see small entry holes on the stem with dry frass at the opening, and in young plants the central shoot dies while the outer leaves stay green, which is called deadheart. Split a stem lengthwise and you find the tunnel.\n\nThe practical difference: fall armyworm can be reached by directing a spray into the whorl. A borer already inside the stem cannot be reached at all, so control there is about timing and destroying residues, not spraying."),
    ]),

    ("faw_vs_armyworm", "compare", [
        ("Is fall armyworm the same as the armyworm that moves in swarms?",
         "They are different pests with similar names, and the control differs.\n\n**Fall armyworm** stays with the crop it hatched on. It feeds inside the maize whorl, is present in the field more or less continuously once established, and is the one that leaves frass in the funnel.\n\n**African armyworm** moves in large migrating groups, strips foliage from cereals and pasture, and appears suddenly in outbreaks after certain rainfall patterns. It feeds openly on the leaves rather than hiding in the whorl.\n\nIf the damage is inside the funnel with frass, treat it as fall armyworm. If large numbers appeared overnight and are eating leaves in the open, report it, because outbreaks are managed at district level and your extension service will want to know."),
    ]),

    ("faw_scouting", "howto", [
        ("How often should I check my maize for fall armyworm?",
         "Scout weekly from emergence until the crop tassels, and pay most attention in the first few weeks, because that is when the pest costs you real yield.\n\nHow to do it properly:\n\n1. Walk a W or X pattern across the field rather than checking the edge, since infestation is patchy.\n2. Open the whorl on about twenty plants spread over that path.\n3. Count how many of the twenty have live larvae or fresh moist frass. Old dry damage with no larvae does not count.\n4. Go early morning or late evening. In the heat of the day the larvae sit deep and you will miss them.\n\nThat count is what tells you whether to act. Treating on the sight of a single caterpillar wastes money; ignoring a heavy early infestation costs the crop."),
    ]),

    ("faw_pidgin_confirm", "pidgin", [
        ("Dem tell me say na stem borer dey my maize but the sawdust dey for the middle of the plant, no be for the stem.",
         "If the sawdust dey for the middle of the plant, for inside the funnel where new leaf dey come out, na fall armyworm, no be stem borer.\n\nHow dem take different:\n\n1. Fall armyworm dey chop for inside the whorl, the middle. You go see hole wey scatter for the young leaf, and the frass wey wet dey pack for centre.\n2. Stem borer dey bore enter the stem. You go see small hole for the stem with dry frass for the mouth of the hole, and sometimes the middle shoot go die while the outside leaf still green.\n\nThe reason e matter: fall armyworm, spray fit reach am if you pour am enter the whorl. Borer wey don enter stem, spray no fit reach am at all, so na clearing residue after harvest be the main thing."),
    ]),
]


def as_records() -> list[dict]:
    return [{"messages": [m for u, a in turns
                          for m in ({"role": "user", "content": u},
                                    {"role": "assistant", "content": a})],
             "_meta": {"slice": "gold", "topic": topic, "form": form}}
            for topic, form, turns in PAIRS]
