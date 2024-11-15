periods_map = {
    "The Belle Époque (1900-1914)": list(range(1900,1914 +1)),
    "World War I (1914-1918)": list(range(1914,1918 +1)),
    "The Roaring Twenties (1920-1929)": list(range(1920,1929 +1)),
    "The Great Depression (1929-1939)": list(range(1929,1939 +1)),
    "World War II (1939-1945)": list(range(1939,1946 +1)),
    "The Cold War and McCarthyism (1947-1991)": list(range(1947,1991 +1)),
    "The Civil Rights and Social Equality Struggles (1950s-1970s)": list(range(1950,1970 +1)),
    "The Reagan Years and the Rise of Neoliberalism (1980s)": list(range(1980,1989 +1)),
    "The Post-Cold War and the New World Order (1991-2001)": list(range(1991,2001 +1)),
    "The 9/11 Attacks and the War on Terrorism (2001-present)": list(range(2001,2024 +1)),
}

# !!! each year currently can have only one period and not multiple
def inverse_dict():
    """inversing the dictionnary"""
    periods_map_inverse_ = {}
    for key, value_list in periods_map.items():
        for item in value_list:
            periods_map_inverse_[item] = key
    return periods_map_inverse_

periods_map_inverse = inverse_dict()