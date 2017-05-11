class Resources:
    """
    Labels used in manual data labeling
    """

    @staticmethod
    def topics():
        """
        Topics label
        Labels picked and annotated by takeLab ( http://takelab.fer.hr )
        """

        return {
            "mobilne usluge": 0,
            "fiksne usluge": 1,
            "brzina interneta": 2,
            "zaposlenici": 3,
            "rjesavanje problema": 4,
            "dijalog": 5,
            "zadovoljstvo opcenito": 6,
            "promocija": 7,
            "usluge": 8,
            "SIM kartica": 9,
            "varanje": 10,
            "signal": 11,
            "kvar": 12,
            "aplikacije": 13,
            "TV": 14,
            "opcenito": 15,
            "paket": 16,
            "tarifa": 17,
            "opcenito podrska": 18,
            "proces": 19,
            "uredaj (HW)": 20,
            "racun": 21,
            "prodajni kanal": 22,
            "dugogodisnji korisnici": 23,
            "odnos opcenito": 24,
            "prituzba": 25,
            "raskid ugovora": 26,
            "nacin obracanja": 27,
            "WiFi": 28,
            "roaming": 29,
            "reklame": 30,
            "ostalo": 31,
            "None": 32,
        }

    @staticmethod
    def emotions():
        """
        Emotion label
        Labels picked and annotated by takeLab ( http://takelab.fer.hr )
        """

        return {
            "None": 0,
            "interest": 1,
            "irritation": 2,
            "friendliness": 3,
            "anger": 4,
            "disappointment": 5,
            "disbelief": 6,
            "excitement": 7,
            "gratitude": 8,
            "helplessness": 9,
            "impatiance": 10,
        }

    @staticmethod
    def speech_acts():
        """
        Speech acts label
        Labels picked and annotated by takeLab ( http://takelab.fer.hr )
        """

        return {
            "asking": 0,
            "insulting/blaming": 1,
            "stating": 2,
            "requesting": 3,
            "warning": 4,
            "apologizing": 5,
            "hoping": 6,
            "praising": 7,
            "rest": 8
        }

    @staticmethod
    def sentiment():
        """
        Sentiment label
        """

        return {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }

    @staticmethod
    def binary_sentiment():
        """
        Binary sentiment label
        Neutral and positive labels are merged together
        """

        return {
            "negative": 0,
            "neutral": 1,
            "positive": 1
        }

    @staticmethod
    def stop_words():
        """
        Croatian stop words
        Taken from : https://github.com/6/stopwords-json/blob/master/dist/hr.json
        """

        return ["a", "ako", "ali", "bi", "bih", "bila", "bili", "bilo", "bio", "bismo", "biste", "biti",
                "bumo", "da",
                "do", "duž", "ga", "hoće", "hoćemo", "hoćete", "hoćeš", "hoću", "i", "iako", "ih", "ili",
                "iz", "ja",
                "je", "jedna", "jedne", "jedno", "jer", "jesam", "jesi", "jesmo", "jest", "jeste", "jesu",
                "jim",
                "joj", "još", "ju", "kada", "kako", "kao", "koja", "koje", "koji", "kojima", "koju", "kroz",
                "li",
                "me", "mene", "meni", "mi", "mimo", "moj", "moja", "moje", "mu", "na", "nad", "nakon", "nam",
                "nama",
                "nas", "naš", "naša", "naše", "našeg", "ne", "nego", "neka", "neki", "nekog", "neku", "nema",
                "netko",
                "neće", "nećemo", "nećete", "nećeš", "neću", "nešto", "ni", "nije", "nikoga", "nikoje",
                "nikoju",
                "nisam", "nisi", "nismo", "niste", "nisu", "njega", "njegov", "njegova", "njegovo", "njemu",
                "njezin",
                "njezina", "njezino", "njih", "njihov", "njihova", "njihovo", "njim", "njima", "njoj", "nju",
                "no",
                "o", "od", "odmah", "on", "ona", "oni", "ono", "ova", "pa", "pak", "po", "pod", "pored",
                "prije", "s",
                "sa", "sam", "samo", "se", "sebe", "sebi", "si", "smo", "ste", "su", "sve", "svi", "svog",
                "svoj",
                "svoja", "svoje", "svom", "ta", "tada", "taj", "tako", "te", "tebe", "tebi", "ti", "to",
                "toj", "tome",
                "tu", "tvoj", "tvoja", "tvoje", "u", "uz", "vam", "vama", "vas", "vaš", "vaša", "vaše",
                "već", "vi",
                "vrlo", "za", "zar", "će", "ćemo", "ćete", "ćeš", "ću", "što"]
