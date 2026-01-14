def fin_fit_disambiguation(text):

    '''
    The program assumes input where the text string has already been identified as Finnish,
    meaning final_prediction = "fin" by some external/off-the-shelf model. When used in a crawler,
    it populates variables for relative character frequencies and occurrences of some lexical items,
    as well as overwrites final_prediction with "fin" or "fit" based on a decision rule.
    '''

    text_lower = text.lower()
    length = len(text_lower)

    # Keeping final_prediction as "fin" in case no string in input
    if length == 0:
        return {
            "length": 0,
            "count_d": 0,
            "rel_freq_d": 0.0,
            "count_h": 0,
            "rel_freq_h": 0.0,
            "count_ette": 0,
            "count_oon": 0,
            "count_mie": 0,
            "count_sie": 0,
            "final_prediction": "fin",
        }

    # Relative character frequencies
    count_d = text_lower.count("d")
    rel_freq_d = count_d / length

    count_h = text_lower.count("h")
    rel_freq_h = count_h / length

    # Lexical markers for meänkieli
    count_ette = text_lower.count(" ette ")
    count_oon = text_lower.count(" oon ")
    count_mie = text_lower.count(" mie ")
    count_sie = text_lower.count(" sie ")

    # Rule used for decision
    if any(x > 0 for x in (count_ette, count_oon, count_mie, count_sie)):  # Change this?
        final_prediction = "fit"   # Meänkieli
    else:
        final_prediction = "fin"   # Finnish

    return {
        "length": length,
        "count_d": count_d,
        "rel_freq_d": rel_freq_d,
        "count_h": count_h,
        "rel_freq_h": rel_freq_h,
        "count_ette": count_ette,
        "count_oon": count_oon,
        "count_mie": count_mie,
        "count_sie": count_sie,
        "final_prediction": final_prediction,
    }

# Run only if run as a main program
if __name__ == "__main__":
    sample_fin = "Suomen kielen hallintoalue Luulajan kunta kuuluu 1. helmikuuta 2013 lähtien Suomen kielen hallintoalueeseen ja palvelemme kuntalaisiamme myös suomen kielellä. Noin 13 000 Luulajalaisella on suomalaiset juuret ja heillä on oikeus tiettyihin kunnan palveluihin suomeksi. Oikeus asioida suomen kielellä Ruotsinsuomalaisena kuntalaisena sinulla on oikeus asioida suomen kielellä kunnan eri virastoissa. Sinulla on myös oikeus saada kirjallinen päätös sinua koskevassa asiassa käännettynä suomen kielelle. Sitä sinun tulee itse pyytää. Kunnan asiakaspalvelussa on suomenkielisiä työntekijöitä, jotka auttavat ja neuvovat suomeksi. Kaikissa asioissa voit olla ensin yhteydessä asiakaspalveluun. Pyydä suomenkielistä palvelua. Asiakaspalvelu neuvoo sinua ja tarvittaessa ohjaa eteenpäin sekä yrittää löytää sinulle suomenkielisen käsittelijän."
    sample_fit = "Luulajanehotus Luulajanehotus (Luleåförslaget) oon palvelu ette sie saatat vaikuttaa Luulajan kehitystä. Anna ehotus ja jos sinun ehotus saapii kylliksi monta ääntä niin se otethaan ylös poliittisesti. Kunkas tämä tehhään? Sie jätät sinun ehotuksen lokkaamalla sisäle e-lekitimasuunin kautta ja selostat mitä toimenpitoa sie halvaat ette kunta tekkee. Katto niitä ehotuksia jokka oon tulheet sisäle kunthaan. Halvaaks jättää uuen ehotuksen? Lue lissää (ruottiksi) Käyttäintymisohjeet Luulajanehotukselle Vain semmosia ehotuksia jokka kuuluvat Luulajan kunna vastuualuheile mennee ottaa ylös. Ehotuksia jokka jo tutkithaan/oon käynissä jatkuvassa olevassa prosesissa ei tulla ottamhaan ylös elikkä jo kunta oon käsitelly samansorttista asiata viimisennä kahen vuen aikana. Ehotus joka saapii 77 ääntä kolmen kuukauen sisälä mennee jatkuvhaan käsittelhyyn. Ei yhthään ehotusta joka sisältää; syrjintää, uhkaa, laitonta toimintaa elikkä sen semmosta tulhaan käsittelheen."
    result_fin = fin_fit_disambiguation(sample_fin)
    result_fit = fin_fit_disambiguation(sample_fit)
    print("\nRESULTS:\n")
    print(f"Prediction results for Finnish text: {result_fin}")
    print("---------------")
    print(f"Prediction results for Meänkieli text: {result_fit}\n")
    #for k, v in result.items():
    #    print(f"{k}: {v}")