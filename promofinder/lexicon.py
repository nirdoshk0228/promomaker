"""Cue lexicons (English stage directions + Hinglish/Hindi dialogue) used by the heuristic scorer.

These encode general promo-editing craft — what makes a scrolling viewer stop, and what makes a clip feel
unfinished — not anything specific to one show. Each entry is a regex fragment matched case-insensitively
on word boundaries.
"""
import re


def _rx(words):
    return re.compile(r"(?<![a-z])(" + "|".join(words) + r")(?![a-z])", re.I)


# Physical shock / violence / danger — the strongest scroll-stoppers
PHYSICAL = _rx([
    r"slap\w*", r"thappad", r"punch\w*", r"swings? (at|across|into)", r"strikes?", r"hits?", r"beats?", r"attacks?", r"kick\w*", r"shove\w*", r"push(es|ed)?", r"yank\w*", r"grab\w*",
    r"throat", r"chok\w*", r"strangl\w*", r"tackl\w*", r"collaps\w*", r"faint\w*", r"falls?", r"fell",
    r"gun", r"revolver", r"pistol", r"shot fired", r"goli", r"knife", r"dagger", r"blade", r"whip\w*",
    r"fire", r"flames?", r"burning", r"catch(es|ing)? fire", r"erupts?", r"petrol", r"match(es)?", r"blood\w*", r"poison\w*", r"zeher", r"foam",
    r"handcuff\w*", r"cuffs", r"kidnap\w*", r"sack over", r"drown\w*", r"underwater", r"splash\w*",
    r"smash\w*", r"shatter\w*", r"crash\w*", r"bursts? open", r"burst in", r"slams?", r"rips?", r"tears? off",
    r"hostage", r"pins?", r"stab\w*", r"kiss\w*", r"lunges?", r"hurls?", r"flings?",
])

# Reveals and identity twists
REVEAL = _rx([
    r"reveal\w*", r"unveil\w*", r"veil falls", r"sheet slides off", r"recognis\w*", r"recogniz\w*",
    r"birthmark", r"portrait", r"has \w+'s (own )?face", r"same (mark|face|crest)",
    r"zinda", r"asli", r"sach", r"pehchaan\w*", r"kokh", r"janmi", r"dna", r"report", r"vasiyat", r"waaris",
    r"rajkumari", r"maharani", r"beti", r"poti", r"qatil", r"qubool", r"likhaawat", r"tum\?!", r"aap\?!",
    r"main hoon", r"kaun ho", r"kaun hai", r"hand clamps", r"a hand grabs",
])

# Threats and stakes in dialogue
THREAT = _rx([
    r"maar\w*", r"jaan", r"laash\w*", r"arthi", r"khatam", r"hamesha ke liye", r"keemat", r"chukaa\w*",
    r"barbaad", r"neelaam", r"sadak pe", r"zubaan kaat", r"goli", r"intezaam", r"le doob\w*", r"nahi bach\w*",
    r"kill", r"die", r"dead", r"warna", r"agli baar", r"zinda nahi", r"aag mein",
])

STRONG_EMOTIONS = {
    "furious", "screaming", "shouting", "venomous", "menacing", "deadly", "hissing", "threatening", "cruel",
    "wild", "shocked", "stunned", "terrified", "panicked", "desperate", "sobbing", "breaking", "choking",
    "snarling", "savage", "blazing", "fierce", "seething", "shrill", "frightened", "gasping", "trembling",
    "pleading", "crying out", "tearful", "breathless", "deadly calm", "chilling", "icy", "cold fury", "spiteful",
    "mocking", "contemptuous", "triumphant", "gleeful", "leering", "drunk", "bitter", "furious, crying",
}
CALM_EMOTIONS = {
    "calm", "gentle", "soft", "polite", "warm", "smiling", "tender", "measured", "technical", "professional",
    "formal", "bored", "flat", "casual", "light", "kind", "helpful", "reassuring", "sweet", "curious", "relieved",
    "analytical", "brisk", "dry", "proud", "encouraging", "wistful", "childlike", "quiet", "even",
}

# Sound / camera cues that writers use to signal an unresolved beat
CLIFF_CUES = _rx([
    r"cliffhanger", r"ominous", r"drone", r"sting", r"string (hit|stab|swell|sting)", r"stab", r"heartbeat",
    r"cut to black", r"silence", r"freez\w*", r"frozen", r"draining", r"pales?", r"widen\w*", r"eye pressed",
    r"shadow", r"silhouette", r"watching", r"listening", r"behind (her|him)", r"bursts? open", r"cocking",
    r"barrel", r"lit match", r"falling toward", r"phone ringing", r"unanswered", r"disconnect", r"stops dead",
    r"single high piano note", r"holding on a single note", r"slowly turning", r"rising",
])

# Things that tie a clip off neatly (bad for an ending)
RESOLUTION = _rx([
    r"hug\w*", r"embrac\w*", r"applause", r"cheer\w*", r"laugh\w*", r"smil\w*", r"relieved", r"relief",
    r"celebrat\w*", r"thank", r"shukriya", r"dhanyavaad", r"sab theek", r"theek hai", r"khush", r"proud",
    r"kisses (her|his) forehead", r"peaceful", r"safe", r"sleeps?", r"asleep", r"nods slowly",
])

# Exposition / procedural content (weak hooks, bad promos)
EXPOSITION = _rx([
    r"crore", r"lakh", r"pratishat", r"percent", r"markers?", r"contract", r"karz", r"loan", r"file",
    r"records?", r"panjeek\w*", r"registered", r"aarop", r"giraftaar\w*", r"dhokhadhadi", r"bayaan",
    r"courier", r"survey", r"map", r"logistics", r"technical", r"formal", r"legal", r"report mein",
    r"company", r"deal", r"bill", r"jama", r"hisaab",
])

# References to things a new viewer has not seen
BACK_REFERENCE = _rx([
    r"pichle", r"kal raat", r"us raat", r"us din", r"pehle (baoli|wala|jaisa)", r"jo tumne", r"jo aapne",
    r"yaad hai", r"bhool gayi", r"bhool gaye", r"wahi (payal|nishaan|ladki)", r"phir se", r"as before",
    r"earlier", r"previous", r"follows", r"again",
])

DREAM = _rx([r"nightmare", r"asleep", r"dream\w*", r"sapna", r"wakes?", r"jerking awake"])
FLASHBACK = _rx([r"flashback", r"flash insert", r"desaturated", r"red-tinted"])
ESTABLISHING = _rx([r"aerial", r"establishing", r"wide shot of (a|the) (grand|empty|lonely|quiet|dim|sunlit)",
                    r"banner", r"exterior"])
COMEDY = _rx([r"comic", r"gag", r"jalebi", r"gossip\w*", r"thumbs-up", r"giant ladle", r"rolling pin"])


def count(rx, text: str) -> int:
    return len(rx.findall(text or ""))


def emotion_energy(emotions: list[str]) -> float:
    """+1 for strongly charged delivery, -0.5 for calm/neutral delivery, 0 otherwise (averaged)."""
    if not emotions:
        return 0.0
    total = 0.0
    for e in emotions:
        el = e.lower().strip()
        if el in STRONG_EMOTIONS:
            total += 1.0
        elif el in CALM_EMOTIONS:
            total -= 0.5
    return max(-1.0, min(1.0, total / max(1, len(emotions)) * 1.5))
