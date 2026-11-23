"""Natural English & Filipino (Tagalog) communication translator, autocomplete, and listener matcher."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

# Display & typo normalization for raw FSL-105 labels.csv entries
LABEL_DISPLAY_FIXES: Dict[str, str] = {
    "IM FINE": "I'M FINE",
    "YOURE WELCOME": "YOU'RE WELCOME",
    "DON'T UNDERSTAND": "DON'T UNDERSTAND",
    "WEELCHAIR PERSON": "WHEELCHAIR PERSON",
    "NINE": "NINE",
}


def clean_fsl_label(raw_label: str) -> str:
    """Normalize whitespace, smart apostrophes, and known typos in FSL-105 labels."""
    norm = re.sub(r"\s+", " ", str(raw_label).replace("’", "'").strip()).upper()
    return LABEL_DISPLAY_FIXES.get(norm, norm)


# Complete 105-entry bilingual communication dictionary keyed by clean uppercase label.
# Each entry provides:
#   en: concise phrase form for multi-sign composition
#   en_sentence: natural standalone sentence for a single sign
#   fil: concise Filipino/Tagalog phrase form
#   fil_sentence: natural standalone Filipino sentence
#   category: FSL-105 category
#   keywords: extra trigger phrases for hearing listener reply matching
FSL105_LEXICON: Dict[str, Dict[str, Any]] = {
    # GREETING (0-9)
    "GOOD MORNING": {"id": 0, "category": "GREETING", "en": "good morning", "en_sentence": "Good morning!", "fil": "magandang umaga", "fil_sentence": "Magandang umaga po!", "keywords": ["good morning", "morning", "magandang umaga"]},
    "GOOD AFTERNOON": {"id": 1, "category": "GREETING", "en": "good afternoon", "en_sentence": "Good afternoon!", "fil": "magandang hapon", "fil_sentence": "Magandang hapon po!", "keywords": ["good afternoon", "afternoon", "magandang hapon"]},
    "GOOD EVENING": {"id": 2, "category": "GREETING", "en": "good evening", "en_sentence": "Good evening!", "fil": "magandang gabi", "fil_sentence": "Magandang gabi po!", "keywords": ["good evening", "evening", "magandang gabi"]},
    "HELLO": {"id": 3, "category": "GREETING", "en": "hello", "en_sentence": "Hello!", "fil": "hello", "fil_sentence": "Hello po!", "keywords": ["hello", "hi", "kumusta"]},
    "HOW ARE YOU": {"id": 4, "category": "GREETING", "en": "how are you?", "en_sentence": "How are you?", "fil": "kumusta ka?", "fil_sentence": "Kumusta ka po?", "keywords": ["how are you", "kumusta ka", "kamusta"]},
    "I'M FINE": {"id": 5, "category": "GREETING", "en": "I'm fine", "en_sentence": "I'm fine, thank you.", "fil": "mabuti naman ako", "fil_sentence": "Mabuti naman po ako.", "keywords": ["i'm fine", "im fine", "i am fine", "fine", "mabuti"]},
    "NICE TO MEET YOU": {"id": 6, "category": "GREETING", "en": "nice to meet you", "en_sentence": "Nice to meet you!", "fil": "ikinagagalak kitang makilala", "fil_sentence": "Ikinagagalak ko pong makilala kayo!", "keywords": ["nice to meet you", "meet you"]},
    "THANK YOU": {"id": 7, "category": "GREETING", "en": "thank you", "en_sentence": "Thank you!", "fil": "salamat", "fil_sentence": "Maraming salamat po!", "keywords": ["thank you", "thanks", "salamat"]},
    "YOU'RE WELCOME": {"id": 8, "category": "GREETING", "en": "you're welcome", "en_sentence": "You're welcome!", "fil": "walang anuman", "fil_sentence": "Walang anuman po!", "keywords": ["you're welcome", "youre welcome", "welcome", "walang anuman"]},
    "SEE YOU TOMORROW": {"id": 9, "category": "GREETING", "en": "see you tomorrow", "en_sentence": "See you tomorrow!", "fil": "kita tayo bukas", "fil_sentence": "Kita tayo bukas!", "keywords": ["see you tomorrow", "kita tayo bukas"]},

    # SURVIVAL (10-19)
    "UNDERSTAND": {"id": 10, "category": "SURVIVAL", "en": "I understand", "en_sentence": "Yes, I understand.", "fil": "naiintindihan ko", "fil_sentence": "Naiintindihan ko po.", "keywords": ["understand", "naiintindihan", "aintindihan"]},
    "DON'T UNDERSTAND": {"id": 11, "category": "SURVIVAL", "en": "I don't understand", "en_sentence": "I don't understand. Could you please repeat or write it?", "fil": "hindi ko maintindihan", "fil_sentence": "Hindi ko po maintindihan. Pakiulit po.", "keywords": ["don't understand", "dont understand", "do not understand", "hindi maintindihan"]},
    "KNOW": {"id": 12, "category": "SURVIVAL", "en": "I know", "en_sentence": "I know.", "fil": "alam ko", "fil_sentence": "Alam ko po.", "keywords": ["know", "alam"]},
    "DON'T KNOW": {"id": 13, "category": "SURVIVAL", "en": "I don't know", "en_sentence": "I don't know.", "fil": "hindi ko alam", "fil_sentence": "Hindi ko po alam.", "keywords": ["don't know", "dont know", "do not know", "hindi alam"]},
    "NO": {"id": 14, "category": "SURVIVAL", "en": "no", "en_sentence": "No.", "fil": "hindi", "fil_sentence": "Hindi po.", "keywords": ["no", "hindi", "nope"]},
    "YES": {"id": 15, "category": "SURVIVAL", "en": "yes", "en_sentence": "Yes.", "fil": "oo", "fil_sentence": "Opo.", "keywords": ["yes", "oo", "opo", "yeah"]},
    "WRONG": {"id": 16, "category": "SURVIVAL", "en": "that is wrong", "en_sentence": "That is wrong.", "fil": "mali iyon", "fil_sentence": "Mali po iyon.", "keywords": ["wrong", "incorrect", "mali"]},
    "CORRECT": {"id": 17, "category": "SURVIVAL", "en": "that is correct", "en_sentence": "That is correct.", "fil": "tama iyan", "fil_sentence": "Tama po iyan.", "keywords": ["correct", "right", "tama"]},
    "SLOW": {"id": 18, "category": "SURVIVAL", "en": "please slow down", "en_sentence": "Please slow down.", "fil": "pakibagalan po", "fil_sentence": "Pakibagalan po.", "keywords": ["slow", "slow down", "bagal"]},
    "FAST": {"id": 19, "category": "SURVIVAL", "en": "too fast", "en_sentence": "That's too fast.", "fil": "masyadong mabilis", "fil_sentence": "Masyado pong mabilis.", "keywords": ["fast", "quick", "mabilis"]},

    # NUMBER (20-29)
    "ONE": {"id": 20, "category": "NUMBER", "en": "1", "en_sentence": "One (1).", "fil": "isa (1)", "fil_sentence": "Isa (1).", "keywords": ["one", "1", "isa"]},
    "TWO": {"id": 21, "category": "NUMBER", "en": "2", "en_sentence": "Two (2).", "fil": "dalawa (2)", "fil_sentence": "Dalawa (2).", "keywords": ["two", "2", "dalawa"]},
    "THREE": {"id": 22, "category": "NUMBER", "en": "3", "en_sentence": "Three (3).", "fil": "tatlo (3)", "fil_sentence": "Tatlo (3).", "keywords": ["three", "3", "tatlo"]},
    "FOUR": {"id": 23, "category": "NUMBER", "en": "4", "en_sentence": "Four (4).", "fil": "apat (4)", "fil_sentence": "Apat (4).", "keywords": ["four", "4", "apat"]},
    "FIVE": {"id": 24, "category": "NUMBER", "en": "5", "en_sentence": "Five (5).", "fil": "lima (5)", "fil_sentence": "Lima (5).", "keywords": ["five", "5", "lima"]},
    "SIX": {"id": 25, "category": "NUMBER", "en": "6", "en_sentence": "Six (6).", "fil": "anim (6)", "fil_sentence": "Anim (6).", "keywords": ["six", "6", "anim"]},
    "SEVEN": {"id": 26, "category": "NUMBER", "en": "7", "en_sentence": "Seven (7).", "fil": "pito (7)", "fil_sentence": "Pito (7).", "keywords": ["seven", "7", "pito"]},
    "EIGHT": {"id": 27, "category": "NUMBER", "en": "8", "en_sentence": "Eight (8).", "fil": "walo (8)", "fil_sentence": "Walo (8).", "keywords": ["eight", "8", "walo"]},
    "NINE": {"id": 28, "category": "NUMBER", "en": "9", "en_sentence": "Nine (9).", "fil": "siyam (9)", "fil_sentence": "Siyam (9).", "keywords": ["nine", "9", "siyam"]},
    "TEN": {"id": 29, "category": "NUMBER", "en": "10", "en_sentence": "Ten (10).", "fil": "sampu (10)", "fil_sentence": "Sampu (10).", "keywords": ["ten", "10", "sampu"]},

    # CALENDAR (30-41)
    "JANUARY": {"id": 30, "category": "CALENDAR", "en": "January", "en_sentence": "January.", "fil": "Enero", "fil_sentence": "Enero.", "keywords": ["january", "enero"]},
    "FEBRUARY": {"id": 31, "category": "CALENDAR", "en": "February", "en_sentence": "February.", "fil": "Pebrero", "fil_sentence": "Pebrero.", "keywords": ["february", "pebrero"]},
    "MARCH": {"id": 32, "category": "CALENDAR", "en": "March", "en_sentence": "March.", "fil": "Marso", "fil_sentence": "Marso.", "keywords": ["march", "marso"]},
    "APRIL": {"id": 33, "category": "CALENDAR", "en": "April", "en_sentence": "April.", "fil": "Abril", "fil_sentence": "Abril.", "keywords": ["april", "abril"]},
    "MAY": {"id": 34, "category": "CALENDAR", "en": "May", "en_sentence": "May.", "fil": "Mayo", "fil_sentence": "Mayo.", "keywords": ["mayo"]},
    "JUNE": {"id": 35, "category": "CALENDAR", "en": "June", "en_sentence": "June.", "fil": "Hunyo", "fil_sentence": "Hunyo.", "keywords": ["june", "hunyo"]},
    "JULY": {"id": 36, "category": "CALENDAR", "en": "July", "en_sentence": "July.", "fil": "Hulyo", "fil_sentence": "Hulyo.", "keywords": ["july", "hulyo"]},
    "AUGUST": {"id": 37, "category": "CALENDAR", "en": "August", "en_sentence": "August.", "fil": "Agosto", "fil_sentence": "Agosto.", "keywords": ["august", "agosto"]},
    "SEPTEMBER": {"id": 38, "category": "CALENDAR", "en": "September", "en_sentence": "September.", "fil": "Setyembre", "fil_sentence": "Setyembre.", "keywords": ["september", "setyembre"]},
    "OCTOBER": {"id": 39, "category": "CALENDAR", "en": "October", "en_sentence": "October.", "fil": "Oktubre", "fil_sentence": "Oktubre.", "keywords": ["october", "oktubre"]},
    "NOVEMBER": {"id": 40, "category": "CALENDAR", "en": "November", "en_sentence": "November.", "fil": "Nobyembre", "fil_sentence": "Nobyembre.", "keywords": ["november", "nobyembre"]},
    "DECEMBER": {"id": 41, "category": "CALENDAR", "en": "December", "en_sentence": "December.", "fil": "Disyembre", "fil_sentence": "Disyembre.", "keywords": ["december", "disyembre"]},

    # DAYS (42-51)
    "MONDAY": {"id": 42, "category": "DAYS", "en": "Monday", "en_sentence": "Monday.", "fil": "Lunes", "fil_sentence": "Lunes.", "keywords": ["monday", "lunes"]},
    "TUESDAY": {"id": 43, "category": "DAYS", "en": "Tuesday", "en_sentence": "Tuesday.", "fil": "Martes", "fil_sentence": "Martes.", "keywords": ["tuesday", "martes"]},
    "WEDNESDAY": {"id": 44, "category": "DAYS", "en": "Wednesday", "en_sentence": "Wednesday.", "fil": "Miyerkules", "fil_sentence": "Miyerkules.", "keywords": ["wednesday", "miyerkules"]},
    "THURSDAY": {"id": 45, "category": "DAYS", "en": "Thursday", "en_sentence": "Thursday.", "fil": "Huwebes", "fil_sentence": "Huwebes.", "keywords": ["thursday", "huwebes"]},
    "FRIDAY": {"id": 46, "category": "DAYS", "en": "Friday", "en_sentence": "Friday.", "fil": "Biyernes", "fil_sentence": "Biyernes.", "keywords": ["friday", "biyernes"]},
    "SATURDAY": {"id": 47, "category": "DAYS", "en": "Saturday", "en_sentence": "Saturday.", "fil": "Sabado", "fil_sentence": "Sabado.", "keywords": ["saturday", "sabado"]},
    "SUNDAY": {"id": 48, "category": "DAYS", "en": "Sunday", "en_sentence": "Sunday.", "fil": "Linggo", "fil_sentence": "Linggo.", "keywords": ["sunday", "linggo"]},
    "TODAY": {"id": 49, "category": "DAYS", "en": "today", "en_sentence": "Today.", "fil": "ngayon", "fil_sentence": "Ngayon.", "keywords": ["today", "ngayon"]},
    "TOMORROW": {"id": 50, "category": "DAYS", "en": "tomorrow", "en_sentence": "Tomorrow.", "fil": "bukas", "fil_sentence": "Bukas.", "keywords": ["tomorrow", "bukas"]},
    "YESTERDAY": {"id": 51, "category": "DAYS", "en": "yesterday", "en_sentence": "Yesterday.", "fil": "kahapon", "fil_sentence": "Kahapon.", "keywords": ["yesterday", "kahapon"]},

    # FAMILY (52-61)
    "FATHER": {"id": 52, "category": "FAMILY", "en": "father", "en_sentence": "My father.", "fil": "tatay", "fil_sentence": "Ang tatay ko.", "keywords": ["father", "dad", "tatay", "ama"]},
    "MOTHER": {"id": 53, "category": "FAMILY", "en": "mother", "en_sentence": "My mother.", "fil": "nanay", "fil_sentence": "Ang nanay ko.", "keywords": ["mother", "mom", "nanay", "ina"]},
    "SON": {"id": 54, "category": "FAMILY", "en": "son", "en_sentence": "My son.", "fil": "anak na lalaki", "fil_sentence": "Ang anak kong lalaki.", "keywords": ["son", "anak na lalaki"]},
    "DAUGHTER": {"id": 55, "category": "FAMILY", "en": "daughter", "en_sentence": "My daughter.", "fil": "anak na babae", "fil_sentence": "Ang anak kong babae.", "keywords": ["daughter", "anak na babae"]},
    "GRANDFATHER": {"id": 56, "category": "FAMILY", "en": "grandfather", "en_sentence": "My grandfather.", "fil": "lolo", "fil_sentence": "Ang lolo ko.", "keywords": ["grandfather", "grandpa", "lolo"]},
    "GRANDMOTHER": {"id": 57, "category": "FAMILY", "en": "grandmother", "en_sentence": "My grandmother.", "fil": "lola", "fil_sentence": "Ang lola ko.", "keywords": ["grandmother", "grandma", "lola"]},
    "UNCLE": {"id": 58, "category": "FAMILY", "en": "uncle", "en_sentence": "My uncle.", "fil": "tito", "fil_sentence": "Ang tito ko.", "keywords": ["uncle", "tito"]},
    "AUNTIE": {"id": 59, "category": "FAMILY", "en": "auntie", "en_sentence": "My auntie.", "fil": "tita", "fil_sentence": "Ang tita ko.", "keywords": ["auntie", "aunt", "tita"]},
    "COUSIN": {"id": 60, "category": "FAMILY", "en": "cousin", "en_sentence": "My cousin.", "fil": "pinsan", "fil_sentence": "Ang pinsan ko.", "keywords": ["cousin", "pinsan"]},
    "PARENTS": {"id": 61, "category": "FAMILY", "en": "parents", "en_sentence": "My parents.", "fil": "mga magulang", "fil_sentence": "Ang mga magulang ko.", "keywords": ["parents", "magulang"]},

    # RELATIONSHIPS (62-71)
    "BOY": {"id": 62, "category": "RELATIONSHIPS", "en": "boy", "en_sentence": "Boy.", "fil": "batang lalaki", "fil_sentence": "Batang lalaki.", "keywords": ["boy", "batang lalaki"]},
    "GIRL": {"id": 63, "category": "RELATIONSHIPS", "en": "girl", "en_sentence": "Girl.", "fil": "batang babae", "fil_sentence": "Batang babae.", "keywords": ["girl", "batang babae"]},
    "MAN": {"id": 64, "category": "RELATIONSHIPS", "en": "man", "en_sentence": "Man.", "fil": "lalaki", "fil_sentence": "Lalaki.", "keywords": ["man", "lalaki"]},
    "WOMAN": {"id": 65, "category": "RELATIONSHIPS", "en": "woman", "en_sentence": "Woman.", "fil": "babae", "fil_sentence": "Babae.", "keywords": ["woman", "babae"]},
    "DEAF": {"id": 66, "category": "RELATIONSHIPS", "en": "I am Deaf", "en_sentence": "I am Deaf.", "fil": "ako ay Bingi", "fil_sentence": "Ako po ay Bingi.", "keywords": ["deaf", "bingi"]},
    "HARD OF HEARING": {"id": 67, "category": "RELATIONSHIPS", "en": "I am hard of hearing", "en_sentence": "I am hard of hearing.", "fil": "mahina ang pandinig ko", "fil_sentence": "Mahina po ang pandinig ko.", "keywords": ["hard of hearing", "hearing"]},
    "WHEELCHAIR PERSON": {"id": 68, "category": "RELATIONSHIPS", "en": "wheelchair user", "en_sentence": "Wheelchair user.", "fil": "gumagamit ng wheelchair", "fil_sentence": "Gumagamit po ng wheelchair.", "keywords": ["wheelchair", "wheelchair person"]},
    "BLIND": {"id": 69, "category": "RELATIONSHIPS", "en": "blind", "en_sentence": "Blind.", "fil": "bulag", "fil_sentence": "Bulag.", "keywords": ["blind", "bulag"]},
    "DEAF BLIND": {"id": 70, "category": "RELATIONSHIPS", "en": "deaf-blind", "en_sentence": "Deaf-blind.", "fil": "bingi at bulag", "fil_sentence": "Bingi at bulag.", "keywords": ["deaf blind", "deaf-blind"]},
    "MARRIED": {"id": 71, "category": "RELATIONSHIPS", "en": "married", "en_sentence": "Married.", "fil": "kasal", "fil_sentence": "Kasal.", "keywords": ["married", "kasal"]},

    # COLOR (72-84)
    "BLUE": {"id": 72, "category": "COLOR", "en": "blue", "en_sentence": "Blue.", "fil": "asul", "fil_sentence": "Asul.", "keywords": ["blue", "asul"]},
    "GREEN": {"id": 73, "category": "COLOR", "en": "green", "en_sentence": "Green.", "fil": "berde", "fil_sentence": "Berde.", "keywords": ["green", "berde"]},
    "RED": {"id": 74, "category": "COLOR", "en": "red", "en_sentence": "Red.", "fil": "pula", "fil_sentence": "Pula.", "keywords": ["red", "pula"]},
    "BROWN": {"id": 75, "category": "COLOR", "en": "brown", "en_sentence": "Brown.", "fil": "kayumanggi", "fil_sentence": "Kayumanggi.", "keywords": ["brown", "kayumanggi"]},
    "BLACK": {"id": 76, "category": "COLOR", "en": "black", "en_sentence": "Black.", "fil": "itim", "fil_sentence": "Itim.", "keywords": ["black", "itim"]},
    "WHITE": {"id": 77, "category": "COLOR", "en": "white", "en_sentence": "White.", "fil": "puti", "fil_sentence": "Puti.", "keywords": ["white", "puti"]},
    "YELLOW": {"id": 78, "category": "COLOR", "en": "yellow", "en_sentence": "Yellow.", "fil": "dilaw", "fil_sentence": "Dilaw.", "keywords": ["yellow", "dilaw"]},
    "ORANGE": {"id": 79, "category": "COLOR", "en": "orange", "en_sentence": "Orange.", "fil": "kahel", "fil_sentence": "Kahel.", "keywords": ["orange", "kahel"]},
    "GRAY": {"id": 80, "category": "COLOR", "en": "gray", "en_sentence": "Gray.", "fil": "abo", "fil_sentence": "Kulay abo.", "keywords": ["gray", "grey", "abo"]},
    "PINK": {"id": 81, "category": "COLOR", "en": "pink", "en_sentence": "Pink.", "fil": "rosas", "fil_sentence": "Rosas.", "keywords": ["pink", "rosas"]},
    "VIOLET": {"id": 82, "category": "COLOR", "en": "violet", "en_sentence": "Violet.", "fil": "lila", "fil_sentence": "Lila.", "keywords": ["violet", "purple", "lila"]},
    "LIGHT": {"id": 83, "category": "COLOR", "en": "light", "en_sentence": "Light color.", "fil": "maliwanag", "fil_sentence": "Maliwanag na kulay.", "keywords": ["light", "maliwanag"]},
    "DARK": {"id": 84, "category": "COLOR", "en": "dark", "en_sentence": "Dark color.", "fil": "madilim", "fil_sentence": "Madilim na kulay.", "keywords": ["dark", "madilim"]},

    # FOOD (85-94)
    "BREAD": {"id": 85, "category": "FOOD", "en": "bread", "en_sentence": "Bread, please.", "fil": "tinapay", "fil_sentence": "Tinapay po.", "keywords": ["bread", "tinapay"]},
    "EGG": {"id": 86, "category": "FOOD", "en": "egg", "en_sentence": "Egg, please.", "fil": "itlog", "fil_sentence": "Itlog po.", "keywords": ["egg", "itlog"]},
    "FISH": {"id": 87, "category": "FOOD", "en": "fish", "en_sentence": "Fish, please.", "fil": "isda", "fil_sentence": "Isda po.", "keywords": ["fish", "isda"]},
    "MEAT": {"id": 88, "category": "FOOD", "en": "meat", "en_sentence": "Meat, please.", "fil": "karne", "fil_sentence": "Karne po.", "keywords": ["meat", "karne"]},
    "CHICKEN": {"id": 89, "category": "FOOD", "en": "chicken", "en_sentence": "Chicken, please.", "fil": "manok", "fil_sentence": "Manok po.", "keywords": ["chicken", "manok"]},
    "SPAGHETTI": {"id": 90, "category": "FOOD", "en": "spaghetti", "en_sentence": "Spaghetti, please.", "fil": "spaghetti", "fil_sentence": "Spaghetti po.", "keywords": ["spaghetti", "pasta"]},
    "RICE": {"id": 91, "category": "FOOD", "en": "rice", "en_sentence": "Rice, please.", "fil": "kanin", "fil_sentence": "Kanin po.", "keywords": ["rice", "kanin"]},
    "LONGANISA": {"id": 92, "category": "FOOD", "en": "longganisa", "en_sentence": "Longganisa, please.", "fil": "longganisa", "fil_sentence": "Longganisa po.", "keywords": ["longanisa", "longganisa"]},
    "SHRIMP": {"id": 93, "category": "FOOD", "en": "shrimp", "en_sentence": "Shrimp, please.", "fil": "hipon", "fil_sentence": "Hipon po.", "keywords": ["shrimp", "hipon"]},
    "CRAB": {"id": 94, "category": "FOOD", "en": "crab", "en_sentence": "Crab, please.", "fil": "alimango", "fil_sentence": "Alimango po.", "keywords": ["crab", "alimango"]},

    # DRINK (95-104)
    "HOT": {"id": 95, "category": "DRINK", "en": "hot", "en_sentence": "Hot, please.", "fil": "mainit", "fil_sentence": "Mainit po.", "keywords": ["hot", "mainit"]},
    "COLD": {"id": 96, "category": "DRINK", "en": "cold", "en_sentence": "Cold, please.", "fil": "malamig", "fil_sentence": "Malamig po.", "keywords": ["cold", "malamig"]},
    "JUICE": {"id": 97, "category": "DRINK", "en": "juice", "en_sentence": "Juice, please.", "fil": "juice", "fil_sentence": "Juice po.", "keywords": ["juice"]},
    "MILK": {"id": 98, "category": "DRINK", "en": "milk", "en_sentence": "Milk, please.", "fil": "gatas", "fil_sentence": "Gatas po.", "keywords": ["milk", "gatas"]},
    "COFFEE": {"id": 99, "category": "DRINK", "en": "coffee", "en_sentence": "Coffee, please.", "fil": "kape", "fil_sentence": "Kape po.", "keywords": ["coffee", "kape"]},
    "TEA": {"id": 100, "category": "DRINK", "en": "tea", "en_sentence": "Tea, please.", "fil": "tsaa", "fil_sentence": "Tsaa po.", "keywords": ["tea", "tsaa"]},
    "BEER": {"id": 101, "category": "DRINK", "en": "beer", "en_sentence": "Beer, please.", "fil": "beer", "fil_sentence": "Beer po.", "keywords": ["beer", "serbesa"]},
    "WINE": {"id": 102, "category": "DRINK", "en": "wine", "en_sentence": "Wine, please.", "fil": "alak", "fil_sentence": "Alak po.", "keywords": ["wine", "alak"]},
    "SUGAR": {"id": 103, "category": "DRINK", "en": "with sugar", "en_sentence": "With sugar, please.", "fil": "may asukal", "fil_sentence": "May asukal po.", "keywords": ["sugar", "asukal"]},
    "NO SUGAR": {"id": 104, "category": "DRINK", "en": "no sugar", "en_sentence": "No sugar, please.", "fil": "walang asukal", "fil_sentence": "Walang asukal po.", "keywords": ["no sugar", "without sugar", "walang asukal"]},
}


# Curated everyday English + Filipino + healthcare/pharmacy/campus autocomplete vocabulary
AUTOCOMPLETE_WORDS: tuple[str, ...] = (
    # Healthcare, pharmacy & accessibility (Proposal Section 1.1)
    "HELP", "DOCTOR", "NURSE", "PHARMACY", "MEDICINE", "PRESCRIPTION", "PAIN",
    "FEVER", "HEADACHE", "COUGH", "ALLERGY", "EMERGENCY", "HOSPITAL", "CLINIC",
    "WATER", "APPOINTMENT", "RECEIPT", "PRICE", "HOW MUCH", "WHERE", "WHEN",
    "WHAT", "WHO", "WHY", "PLEASE", "SORRY", "EXCUSE ME", "WAIT", "REPEAT",
    "WRITE", "PAPER", "PEN", "PHONE", "RESTROOM", "ENTRANCE", "EXIT",
    "NAME", "ADDRESS", "ID", "NEED", "WANT", "HAVE", "DO NOT", "CAN YOU",
    "GAMOT", "TULONG", "MASAKIT", "LAGNAT", "UBO", "OSPITAS", "OSPITAL",
    "DOKTOR", "NARS", "BOTIKA", "RESETA", "TUBIG", "MAGKANO", "SAAN", "KAILAN",
    "ANO", "SINO", "BAKIT", "PAKIULIT", "PAKISULAT", "SANDALI", "PANGALAN",
    "SALAMAT", "PAUMANHIN", "BINGI", "MAHAL", "MURA", "BUKAS", "NGAYON",
    "KAHAPON", "UMAGA", "HAPON", "GABI", "KAPE", "KANIN", "TINAPAY", "ITLOG",
    "MANOK", "ISDA", "KARNE", "GATAS", "TSAA", "ASUKAL", "MAINIT", "MALAMIG",
    "OO", "OPO", "HINDI", "TAMA", "MALI", "ALAM", "NAIINTINDIHAN",
    # Everyday English words
    "GOOD", "MORNING", "AFTERNOON", "EVENING", "HELLO", "THANK", "YOU",
    "WELCOME", "SORRY", "UNDERSTAND", "KNOW", "DEAF", "SIGN", "LANGUAGE",
    "SLOW", "FAST", "YES", "NO", "CORRECT", "WRONG", "TODAY", "TOMORROW",
    "YESTERDAY", "FATHER", "MOTHER", "BROTHER", "SISTER", "FAMILY", "FRIEND",
    "COUSIN", "PARENTS", "COFFEE", "TEA", "MILK", "JUICE", "BREAD", "RICE",
    "CHICKEN", "FISH", "MEAT", "EGG", "HOT", "COLD", "SUGAR", "SWEET",
    "BLUE", "GREEN", "RED", "BLACK", "WHITE", "YELLOW", "ORANGE", "PINK",
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
    "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN",
)


def get_lexicon_entry(label: str) -> Dict[str, Any]:
    clean = clean_fsl_label(label)
    if clean in FSL105_LEXICON:
        return FSL105_LEXICON[clean]
    return {
        "id": -1,
        "category": "CUSTOM",
        "en": clean.lower(),
        "en_sentence": clean.capitalize() + ".",
        "fil": clean.lower(),
        "fil_sentence": clean.capitalize() + ".",
        "keywords": [clean.lower()],
    }


def compose_sentence(tokens: Sequence[Any], lang: str = "en") -> str:
    """Convert a sequence of FSL-105 phrase tokens and/or spelled text into a natural sentence."""
    if not tokens:
        return ""

    norm_tokens: List[Dict[str, str]] = []
    for tok in tokens:
        if isinstance(tok, str):
            text = tok.strip()
            if text:
                norm_tokens.append({"type": "text", "label": text})
        elif isinstance(tok, dict):
            ttype = tok.get("type", "phrase")
            lbl = str(tok.get("label", "")).strip()
            if lbl:
                norm_tokens.append({"type": ttype, "label": clean_fsl_label(lbl) if ttype == "phrase" else lbl})

    if not norm_tokens:
        return ""

    # Single phrase sign -> idiomatic standalone sentence
    if len(norm_tokens) == 1 and norm_tokens[0]["type"] == "phrase":
        entry = get_lexicon_entry(norm_tokens[0]["label"])
        return entry["fil_sentence"] if lang == "fil" else entry["en_sentence"]

    # Multi-token intelligent composition
    phrases: List[str] = []
    i = 0
    while i < len(norm_tokens):
        cur = norm_tokens[i]
        if cur["type"] != "phrase":
            # Spelled word or custom text
            word = cur["label"]
            phrases.append(word.lower() if word.isupper() and len(word) > 1 else word)
            i += 1
            continue

        lbl = cur["label"]
        entry = get_lexicon_entry(lbl)
        nxt = norm_tokens[i + 1] if i + 1 < len(norm_tokens) else None
        nxt_lbl = nxt["label"] if (nxt and nxt["type"] == "phrase") else None

        # Pattern: NUMBER + FOOD/DRINK (e.g., TWO + COFFEE -> "two coffee" / "dalawang kape")
        if entry["category"] == "NUMBER" and nxt_lbl and get_lexicon_entry(nxt_lbl)["category"] in {"FOOD", "DRINK"}:
            nxt_entry = get_lexicon_entry(nxt_lbl)
            if lang == "fil":
                num_word = entry["fil"].split(" ")[0]
                phrases.append(f"{num_word} {nxt_entry['fil']}")
            else:
                phrases.append(f"{entry['en']} {nxt_entry['en']}")
            i += 2
            continue

        # Pattern: HOT/COLD + DRINK/FOOD (e.g., HOT + COFFEE -> "hot coffee" / "mainit na kape")
        if lbl in {"HOT", "COLD"} and nxt_lbl and get_lexicon_entry(nxt_lbl)["category"] in {"FOOD", "DRINK"}:
            nxt_entry = get_lexicon_entry(nxt_lbl)
            if lang == "fil":
                phrases.append(f"{entry['fil']} na {nxt_entry['fil']}")
            else:
                phrases.append(f"{entry['en']} {nxt_entry['en']}")
            i += 2
            continue

        # Pattern: DRINK/FOOD + NO SUGAR / SUGAR
        if entry["category"] in {"FOOD", "DRINK"} and nxt_lbl in {"NO SUGAR", "SUGAR"}:
            nxt_entry = get_lexicon_entry(nxt_lbl)
            if lang == "fil":
                phrases.append(f"{entry['fil']} na {nxt_entry['fil']}")
            else:
                phrases.append(f"{entry['en']} ({nxt_entry['en']})")
            i += 2
            continue

        # Pattern: COLOR + LIGHT/DARK or LIGHT/DARK + COLOR
        if lbl in {"LIGHT", "DARK"} and nxt_lbl and get_lexicon_entry(nxt_lbl)["category"] == "COLOR":
            nxt_entry = get_lexicon_entry(nxt_lbl)
            if lang == "fil":
                phrases.append(f"{entry['fil']} na {nxt_entry['fil']}")
            else:
                phrases.append(f"{entry['en']} {nxt_entry['en']}")
            i += 2
            continue

        # Pattern: DEAF / HARD OF HEARING + DON'T UNDERSTAND / SLOW
        if lbl in {"DEAF", "HARD OF HEARING"} and nxt_lbl in {"DON'T UNDERSTAND", "SLOW"}:
            nxt_entry = get_lexicon_entry(nxt_lbl)
            if lang == "fil":
                phrases.append(f"{entry['fil']}, {nxt_entry['fil']}")
            else:
                phrases.append(f"{entry['en']}, {nxt_entry['en']}")
            i += 2
            continue

        # Pattern: TOMORROW / TODAY / YESTERDAY + DAY OF WEEK
        if lbl in {"TODAY", "TOMORROW", "YESTERDAY"} and nxt_lbl and get_lexicon_entry(nxt_lbl)["category"] == "DAYS":
            nxt_entry = get_lexicon_entry(nxt_lbl)
            if lang == "fil":
                phrases.append(f"{entry['fil']}, {nxt_entry['fil']}")
            else:
                phrases.append(f"{entry['en']} ({nxt_entry['en']})")
            i += 2
            continue

        phrases.append(entry["fil"] if lang == "fil" else entry["en"])
        i += 1

    # Join parts gracefully
    joined = ""
    for idx, part in enumerate(phrases):
        if idx == 0:
            joined = part[0].upper() + part[1:]
        else:
            prev_ends_punct = joined.endswith(("?", "!", "."))
            if prev_ends_punct:
                joined += " " + part[0].upper() + part[1:]
            elif any(joined.lower().startswith(g) for g in ("good ", "hello", "magandang ", "kumusta")):
                joined += ", " + part
            else:
                joined += " " + part

    if not joined.endswith((".", "!", "?")):
        if any(t["label"] == "HOW ARE YOU" for t in norm_tokens) and "?" not in joined:
            joined += "?"
        elif any(get_lexicon_entry(t["label"])["category"] in {"FOOD", "DRINK"} for t in norm_tokens if t["type"] == "phrase"):
            joined += " po." if lang == "fil" else ", please."
        else:
            joined += "."
    return joined


def autocomplete_word(prefix: str, limit: int = 6) -> List[str]:
    """Return frequent bilingual/communication word completions matching prefix."""
    clean = re.sub(r"[^A-Za-z]", "", prefix).upper()
    if not clean:
        return []

    exact_or_prefix: List[str] = []
    seen = set()
    for word in AUTOCOMPLETE_WORDS:
        if word.startswith(clean) and word != clean and word not in seen:
            seen.add(word)
            exact_or_prefix.append(word)
            if len(exact_or_prefix) >= limit:
                break
    return exact_or_prefix


def match_listener_signs(text: str) -> List[Dict[str, Any]]:
    """Match words/phrases in a hearing person's reply to available FSL-105 signs."""
    if not text or not text.strip():
        return []

    lowered = re.sub(r"[.,!?;:\"]+", " ", text.lower().replace("’", "'"))
    lowered = re.sub(r"\s+", " ", lowered).strip()

    # Build candidate (keyword, label, id, category) pairs sorted longest keyword first
    candidates = []
    for label, info in FSL105_LEXICON.items():
        for kw in info.get("keywords", []):
            candidates.append((kw.lower(), label, info["id"], info["category"]))
    candidates.sort(key=lambda item: len(item[0]), reverse=True)

    matched: List[Tuple[int, Dict[str, Any]]] = []
    used_ids = set()
    padded = f" {lowered} "

    for kw, label, cid, category in candidates:
        if cid in used_ids:
            continue
        needle = f" {kw} "
        pos = padded.find(needle)
        if pos != -1:
            used_ids.add(cid)
            matched.append((pos, {"id": cid, "label": label, "category": category, "matched_text": kw}))

    matched.sort(key=lambda item: item[0])
    return [item[1] for item in matched]
