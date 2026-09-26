import csv
import math

file_name = "digital_twins"
modo = 1
threshold = 70.0
subscore_threshold = 30.0

MODE_2_COLUMNS = (
    "core__specific_counterpart",
    "core__individualized_data_link",
    "core__simulation_prediction",
)


def convert(file_name, modo, threshold, subscore_threshold):
    if modo not in (1, 2):
        raise ValueError("modo deve ser 1 (fit_score) ou 2 (subscores).")

    with open(f"{file_name}.csv", encoding="utf-8-sig", newline="") as entrada:
        leitor = csv.DictReader(entrada, delimiter=",")
        required = ("fit_score",) if modo == 1 else MODE_2_COLUMNS
        missing = [
            column for column in required if column not in (leitor.fieldnames or ())
        ]
        missing += [
            column
            for column in ("site_name", "site_url")
            if column not in (leitor.fieldnames or ())
        ]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(missing)}")

        selected = []
        for linha in leitor:
            scores = []
            for column in required:
                try:
                    score = float(linha[column])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Subscore inválido para {linha['site_name']}: {column}"
                    ) from exc
                if not math.isfinite(score):
                    raise ValueError(
                        f"Subscore inválido para {linha['site_name']}: {column}"
                    )
                scores.append(score)

            approved = (
                scores[0] > threshold
                if modo == 1
                else all(score >= subscore_threshold for score in scores)
            )
            if approved:
                selected.append((linha["site_name"], linha["site_url"]))

    with open(f"{file_name}_simples.csv", "w", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida)
        escritor.writerow(["Company", "URL"])
        escritor.writerows(selected)

    print(f"Arquivo criado: {file_name}_simples.csv")


if __name__ == "__main__":
    convert(file_name, modo, threshold, subscore_threshold)
