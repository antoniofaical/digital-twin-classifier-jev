import csv

file_name = "digital_twins"
modo = 2
threshold = 70.0
subscore_threshold = 30.0

if modo not in (1, 2):
    raise ValueError("modo deve ser 1 (fit_score) ou 2 (subscores).")

with open(f"{file_name}.csv", encoding="utf-8-sig", newline="") as entrada:
    leitor = csv.DictReader(entrada, delimiter=",")

    subscores = [
        coluna
        for coluna in leitor.fieldnames
        if coluna.startswith(("core__", "aux__"))
        and not coluna.endswith("__flag")
    ]

    if modo == 2 and not subscores:
        raise ValueError("Nenhuma coluna de subscore encontrada.")

    with open(f"{file_name}_simples.csv", "w", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida)
        escritor.writerow(["Company", "URL"])

        for linha in leitor:
            if modo == 1:
                aprovado = float(linha["fit_score"]) > threshold
            else:
                aprovado = all(
                    float(linha[coluna]) >= subscore_threshold
                    for coluna in subscores
                )

            if aprovado:
                escritor.writerow([linha["site_name"], linha["site_url"]])

print(f"Arquivo criado: {file_name}_simples.csv")
