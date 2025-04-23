# attribute-weighting

# rank_plik.py - główny plik
 - Skrypt trenuje RandomForestClassifier, tworzy ranking cech według głębokości pierwszego podziału oraz przeprowadza usuwanie cech w kolejności rankingu.
 - generowany jest plik excel z wynikami
  # UWAGA
    - linia 162: ścieżka do pliku uczącego
    - linia 163: ścieżka do pliku test 1
    - linia 164: ścieżka do pliku test 2

    - linia 167: nazwa pliku z rankigiem jeśli nie podany, to skrypt sam tworzy ranking
    
# custom_ranking.csv - plik z gotowym rankingiem cech
 - usuwane cechy po kolei od góry pliku (uwaga na spacje - BEZ SPACJI)
