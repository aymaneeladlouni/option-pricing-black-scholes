"""
PRICING D'OPTIONS - BLACK-SCHOLES ET MONTE CARLO
Auteur : Aymane El Adlouni

Ce programme calcule le prix d'une option financiere selon deux methodes
complementaires, puis verifie qu'elles convergent vers le meme resultat.

Une option est un contrat donnant le droit, mais pas l'obligation, d'acheter
(call) ou de vendre (put) une action a un prix fixe, a une date future. La
question centrale du pricing est : combien vaut ce droit aujourd'hui ?

Deux approches sont mises en oeuvre :
    1. Black-Scholes : la formule mathematique qui donne le prix theorique exact
    2. Monte Carlo : la simulation de milliers de trajectoires de prix

On calcule aussi les principales sensibilites (les "grecques"), puis on
visualise les trajectoires simulees et la convergence des deux methodes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Parametres de l'option a evaluer.
prix_actuel = 100.0       # Prix actuel de l'action (S)
prix_exercice = 105.0     # Prix d'exercice de l'option (K)
maturite = 1.0            # Temps jusqu'a l'echeance, en annees (T)
taux_sans_risque = 0.03   # Taux sans risque annuel (r)
volatilite = 0.25         # Volatilite annuelle de l'action (sigma)


def black_scholes(S, K, T, r, sigma, type_option="call"):
    """
    Calcule le prix d'une option europeenne par la formule de Black-Scholes.

    La formule repose sur deux termes d1 et d2 qui mesurent, en tenant compte
    du temps et de la volatilite, la position du prix actuel par rapport au
    prix d'exercice. Elle donne le prix theorique exact de l'option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type_option == "call":
        prix = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        prix = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return prix


def grecques(S, K, T, r, sigma, type_option="call"):
    """
    Calcule les principales sensibilites de l'option, appelees les grecques.

    Le Delta mesure de combien varie le prix de l'option quand l'action bouge
    de 1. Le Vega mesure la sensibilite a la volatilite. Le Theta mesure la
    perte de valeur liee au temps qui passe. Ce sont les outils quotidiens
    d'un trader pour gerer et couvrir ses positions.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type_option == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) / 365

    return delta, vega, theta


def monte_carlo(S, K, T, r, sigma, n_simulations=100000, type_option="call"):
    """
    Estime le prix de l'option en simulant des milliers de trajectoires.

    Au lieu d'une formule, on simule un grand nombre de prix possibles de
    l'action a l'echeance, en supposant qu'elle suit un mouvement aleatoire.
    Pour chaque scenario on calcule le gain de l'option, puis on prend la
    moyenne de ces gains, actualisee a aujourd'hui. Plus le nombre de
    simulations est grand, plus l'estimation se rapproche de Black-Scholes.
    """
    # Prix de l'action a l'echeance pour chaque simulation
    hasard = np.random.standard_normal(n_simulations)
    prix_final = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * hasard)

    if type_option == "call":
        gains = np.maximum(prix_final - K, 0)
    else:
        gains = np.maximum(K - prix_final, 0)

    prix = np.exp(-r * T) * np.mean(gains)
    erreur = np.exp(-r * T) * np.std(gains) / np.sqrt(n_simulations)

    return prix, erreur


def simuler_trajectoires(S, T, r, sigma, n_trajectoires=200, n_pas=252):
    """
    Simule l'evolution du prix de l'action jour par jour jusqu'a l'echeance.

    Chaque trajectoire represente un scenario possible d'evolution du cours
    sur une annee (252 jours de bourse). L'ensemble illustre l'incertitude
    sur le prix futur, fondement de la simulation Monte Carlo.
    """
    dt = T / n_pas
    trajectoires = np.zeros((n_trajectoires, n_pas + 1))
    trajectoires[:, 0] = S

    for t in range(1, n_pas + 1):
        hasard = np.random.standard_normal(n_trajectoires)
        trajectoires[:, t] = trajectoires[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * hasard
        )

    return trajectoires


def afficher_resultats():
    """Compare les deux methodes de pricing et affiche les grecques."""
    print("Parametres de l'option")
    print(f"Prix actuel de l'action : {prix_actuel}")
    print(f"Prix d'exercice : {prix_exercice}")
    print(f"Maturite : {maturite} an")
    print(f"Taux sans risque : {taux_sans_risque*100:.1f}%")
    print(f"Volatilite : {volatilite*100:.1f}%")
    print()

    for type_option in ["call", "put"]:
        prix_bs = black_scholes(prix_actuel, prix_exercice, maturite,
                                taux_sans_risque, volatilite, type_option)
        prix_mc, erreur = monte_carlo(prix_actuel, prix_exercice, maturite,
                                      taux_sans_risque, volatilite,
                                      type_option=type_option)
        delta, vega, theta = grecques(prix_actuel, prix_exercice, maturite,
                                      taux_sans_risque, volatilite, type_option)

        print(f"Option {type_option.upper()}")
        print(f"Prix Black-Scholes : {prix_bs:.4f}")
        print(f"Prix Monte Carlo : {prix_mc:.4f} (+/- {erreur:.4f})")
        print(f"Ecart entre les deux methodes : {abs(prix_bs - prix_mc):.4f}")
        print(f"Delta : {delta:.4f}")
        print(f"Vega : {vega:.4f}")
        print(f"Theta : {theta:.4f}")
        print()


def tracer_graphiques():
    """Trace les trajectoires simulees et la convergence Monte Carlo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique 1 : trajectoires simulees du prix de l'action
    trajectoires = simuler_trajectoires(prix_actuel, maturite,
                                        taux_sans_risque, volatilite)
    jours = np.linspace(0, maturite, trajectoires.shape[1])
    for i in range(trajectoires.shape[0]):
        ax1.plot(jours, trajectoires[i], linewidth=0.6, alpha=0.4)
    ax1.axhline(prix_exercice, color="red", linestyle="--",
                label=f"Prix d'exercice ({prix_exercice})")
    ax1.set_xlabel("Temps (annees)")
    ax1.set_ylabel("Prix de l'action")
    ax1.set_title("Trajectoires simulees du prix de l'action")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Graphique 2 : convergence de Monte Carlo vers Black-Scholes
    prix_bs = black_scholes(prix_actuel, prix_exercice, maturite,
                            taux_sans_risque, volatilite, "call")
    tailles = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    prix_mc_liste = []
    for n in tailles:
        prix_mc, _ = monte_carlo(prix_actuel, prix_exercice, maturite,
                                 taux_sans_risque, volatilite, n, "call")
        prix_mc_liste.append(prix_mc)

    ax2.plot(tailles, prix_mc_liste, marker="o", color="#4C72B0",
             label="Monte Carlo")
    ax2.axhline(prix_bs, color="red", linestyle="--",
                label=f"Black-Scholes ({prix_bs:.3f})")
    ax2.set_xscale("log")
    ax2.set_xlabel("Nombre de simulations")
    ax2.set_ylabel("Prix estime de l'option (call)")
    ax2.set_title("Convergence de Monte Carlo vers Black-Scholes")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("option_pricing_resultats.png", dpi=120)
    print("Graphique enregistre : option_pricing_resultats.png")


if __name__ == "__main__":
    np.random.seed(42)
    afficher_resultats()
    tracer_graphiques()
