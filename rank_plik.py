import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split

def parse_tab_file(filepath):
    
    #Parsuje plik .tab o zadanej strukturze i zwraca DataFrame.
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() != ""]
    
    n_attr = int(lines[1].split()[1])
    attributes = []
    for i in range(2, 2 + n_attr):
        parts = lines[i].split()
        attributes.append(parts[0])
    
    n_objects = int(lines[2 + n_attr].split()[1])
    data_lines = lines[3 + n_attr: 3 + n_attr + n_objects]
    data = [line.split(',') for line in data_lines]
    
    df = pd.DataFrame(data, columns=attributes)
    return df

def get_tree_feature_depths(tree):
    
    feature = tree.feature
    depths = {}
    queue = [(0, 0)]  # zaczynamy od korzenia
    while queue:
        node_id, depth = queue.pop(0)
        if feature[node_id] != _tree.TREE_UNDEFINED:
            feat = feature[node_id]
            if feat not in depths:
                depths[feat] = depth
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]
            if left != -1:
                queue.append((left, depth + 1))
            if right != -1:
                queue.append((right, depth + 1))
    return depths

def get_forest_feature_ranking(forest, n_features):
    """
    Dla danego RandomForestClassifier (trenowanego na danych uczących) oblicza dla każdego atrybutu:
      - listę głębokości pierwszego wystąpienia (dla każdego drzewa),
        gdzie jeśli atrybut wystąpi, zapisujemy jego oryginalną głębokość (dla wyświetlenia);
        jeśli nie, zapisujemy "-" (dla wyświetlenia), a do celów obliczeniowych traktujemy to jako 0.
      - Finalną średnią głębokość obliczaną jako:
            (suma oryginalnych głębokości dla drzew, gdzie cecha wystąpiła) / (liczba drzew)
            plus liczba drzew, w których cecha wystąpiła jako korzeń (czyli gdzie głębokość == 0).
      - Minimalną głębokość (z oryginalnych wartości, gdzie 0 pozostaje 0).
      - Wagę = (final_avg) * (liczba drzew, w których atrybut wystąpił).
      
    Wyniki sortujemy wg. wagi rosnąco, a przy remisie wg. minimalnej głębokości (niższa wartość jest lepsza).
    
    Zwraca listę krotek:
      (indeks, lista_poziomow, final_avg, min_depth, waga)
    """
    all_display = {i: [] for i in range(n_features)}
    
    for estimator in forest.estimators_:
        tree = estimator.tree_
        tree_depths = get_tree_feature_depths(tree)
        for i in range(n_features):
            if i in tree_depths:
                all_display[i].append(tree_depths[i])
            else:
                all_display[i].append("-")
    
    ranking = []
    num_trees = len(forest.estimators_)  # np. 5
    for i in range(n_features):
        orig_list = [d if d != "-" else 0 for d in all_display[i]]
        sum_orig = np.sum(orig_list)
        count_present = sum(1 for d in all_display[i] if d != "-")
        count_zeros = sum(1 for d in all_display[i] if d == 0)
        avg_orig = sum_orig / num_trees
        final_avg = avg_orig + count_zeros
        valid = [d for d in all_display[i] if d != "-"]
        if valid:
            min_depth = min(valid)
        else:
            min_depth = "-"
        weight = final_avg * count_present
        ranking.append((i, all_display[i], final_avg, min_depth, weight))
    
    ranking.sort(key=lambda x: (x[4], (x[3] if x[3] != "-" else np.inf)))
    return ranking

def compute_forest_stats(rf):
    """
    Oblicza statystyki lasu:
      - Średnia głębokość (średnia maksymalna głębokość drzew),
      - Średnia liczba węzłów w drzewach.
    """
    depths = [estimator.tree_.max_depth for estimator in rf.estimators_]
    nodes = [estimator.tree_.node_count for estimator in rf.estimators_]
    avg_depth = np.mean(depths)
    avg_nodes = np.mean(nodes)
    return avg_depth, avg_nodes

def ablation_study(train_X, train_y, test_X, test_y, features, removal_order, n_estimators=5, random_state=42):
    """
    Przeprowadza ablation study:
      - Trenowanie modelu na zbiorze treningowym dla danego zestawu cech,
        ocena na zbiorze testowym.
      - removal_order: lista nazw cech w kolejności, w jakiej mają być usuwane.
    Dla każdego etapu obliczamy:
      - Liczbę cech,
      - Dokładność (Accuracy),
      - Średnią głębokość lasu,
      - Średnią liczbę węzłów.
      
    Zwraca DataFrame z kolumnami:
      'Liczba_Cech', 'Accuracy', 'SredniaGlebokoscLasu', 'SredniaLiczbaWezlow'
    """
    results = []
    remaining = list(features)
    
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(train_X, train_y)
    acc = clf.score(test_X, test_y)
    avg_depth, avg_nodes = compute_forest_stats(clf)
    results.append((len(remaining), acc, avg_depth, avg_nodes))
    
    for feat in removal_order:
        if feat in remaining:
            remaining.remove(feat)
        print("Iteracja, usuwam:", feat, " -> remaining:", remaining)            
        if len(remaining) == 0:
            break
        idx = [i for i, f in enumerate(features) if f in remaining]
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(train_X[:, idx], train_y)
        acc = clf.score(test_X[:, idx], test_y)
        avg_depth, avg_nodes = compute_forest_stats(clf)
        results.append((len(remaining), acc, avg_depth, avg_nodes))
    
    ablation_df = pd.DataFrame(results, columns=["Liczba_Cech", "Accuracy", "SredniaGlebokoscLasu", "SredniaLiczbaWezlow"])
    return ablation_df

def load_custom_ranking(filepath, features):
    """
    Wczytuje ranking cech z pliku CSV.
    
    """
    custom_df = pd.read_csv(filepath)
    # Upewnij się, że ranking zawiera tylko cechy znajdujące się w features
    removal_order = [feat for feat in custom_df["NazwaAtrybutu"].tolist() if feat in features]
    return removal_order

def main():
    # Ścieżki do plików:
    
    
    train_filepath = os.path.join("duw", "duw", "FLduwb00008URses.tab") # Plik uczący
    test_filepath1  = os.path.join("duw", "duw", "FT1duwb00008URses.tab")# Pliki testowe: 1
    test_filepath2  = os.path.join("duw", "duw", "FT2duwb00008URses.tab")# Pliki testowe: 2
    
    # Opcjonalnie, ścieżka do pliku z własnym rankingiem (CSV)
    custom_ranking_file = "custom_ranking.csv"  # jeśli plik istnieje, zostanie użyty
    
    print("Trenujemy na pliku:", train_filepath)
    try:
        df_train = parse_tab_file(train_filepath)
    except Exception as e:
        print("Błąd przy parsowaniu pliku treningowego:", train_filepath, e)
        return
    
    print("Testujemy na pliku 1:", test_filepath1)
    try:
        df_test1 = parse_tab_file(test_filepath1)
    except Exception as e:
        print("Błąd przy parsowaniu pliku testowego 1:", test_filepath1, e)
        return
    
    print("Testujemy na pliku 2:", test_filepath2)
    try:
        df_test2 = parse_tab_file(test_filepath2)
    except Exception as e:
        print("Błąd przy parsowaniu pliku testowego 2:", test_filepath2, e)
        return
    
    # Ostatnia kolumna to etykieta, pozostałe to cechy
    features = df_train.columns[:-1]
    target_col = df_train.columns[-1]
    
    for col in features:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test1[col] = pd.to_numeric(df_test1[col], errors='coerce')
        df_test2[col] = pd.to_numeric(df_test2[col], errors='coerce')
    
    le = LabelEncoder()
    df_train[target_col] = le.fit_transform(df_train[target_col])
    df_test1[target_col] = le.transform(df_test1[target_col])
    df_test2[target_col] = le.transform(df_test2[target_col])
    
    X_train = df_train[features].values
    y_train = df_train[target_col].values
    X_test1 = df_test1[features].values
    y_test1 = df_test1[target_col].values
    X_test2 = df_test2[features].values
    y_test2 = df_test2[target_col].values
    
    # Trenujemy model na zbiorze treningowym
    rf_train = RandomForestClassifier(n_estimators=5, random_state=42)
    rf_train.fit(X_train, y_train)
    base_accuracy1 = rf_train.score(X_test1, y_test1)
    base_accuracy2 = rf_train.score(X_test2, y_test2)
    
    # Jeśli plik z własnym rankingiem istnieje, wczytujemy removal_order z niego;
    # w przeciwnym przypadku, obliczamy ranking automatycznie.
    if os.path.exists(custom_ranking_file):
        print("Wczytujemy własny ranking z:", custom_ranking_file)
        removal_order = load_custom_ranking(custom_ranking_file, features)
    else:
        ranking = get_forest_feature_ranking(rf_train, len(features))
        ranking.sort(key=lambda x: (x[4], (x[3] if x[3] != "-" else np.inf)))
        removal_order = [features[r[0]] for r in ranking]
    
    # Przeprowadzamy ablation study dla obu zbiorów testowych (przy użyciu tego samego removal_order)
    ablation_df1 = ablation_study(X_train, y_train, X_test1, y_test1, list(features), removal_order, n_estimators=5, random_state=42)
    ablation_df2 = ablation_study(X_train, y_train, X_test2, y_test2, list(features), removal_order, n_estimators=5, random_state=42)
    
    # Łączymy wyniki ablation study na obu zbiorach testowych – łączymy po kolumnie "Liczba_Cech"
    merged_df = pd.merge(ablation_df1, ablation_df2, on="Liczba_Cech", suffixes=("_test1", "_test2"))
    
    # Dodajemy kolumnę z uśrednioną dokładnością oraz kolumnę z odchyleniem standardowym dokładności
    merged_df["Avg_Accuracy"] = (merged_df["Accuracy_test1"] + merged_df["Accuracy_test2"]) / 2
    merged_df["Std_Accuracy"] = merged_df.apply(lambda row: np.std([row["Accuracy_test1"], row["Accuracy_test2"]]), axis=1)
    
    # Zapisujemy wyniki do jednego arkusza Excel
    output_excel = "merged_results.xlsx"
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        merged_df.to_excel(writer, sheet_name="Ablation", index=False)
    
   
    
    print("Wyniki study zapisano w pliku:", output_excel)

if __name__ == "__main__":
    main()
