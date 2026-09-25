# Startup Theme Adherence

Avalia evidências de sites de startups ou arquivos de texto contra um **perfil
versionado**. O perfil define as perguntas enviadas ao Jev, quais critérios
entram na pontuação e como as respostas são agregadas. O perfil `digital_twin`
é o padrão para sites; outros perfis, como
`profiles/gsd_patient_journey_mapping.json`, usam o mesmo pipeline.

## Instalação

Python 3.11 ou superior:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests orchestrator.py fit.py
python -m ruff format --check src tests orchestrator.py fit.py
```

Para extrair PDFs, instale também `.[pdf]`. O extrator aceita HTML e texto sem
essa dependência. `orchestrator.py` e `fit.py` são entradas compatíveis para
uso no checkout; o pacote instalado fornece `startup-adherence` e
`startup-fit`.

## Avaliação de sites

Configure `sites.json` como uma lista de `{"name": "empresa", "url":
"https://empresa.example/"}`. Os modos funcionam com `--sites-file`, `--site`,
`--workers` e `--evidence-root`. Use `--profile` para qualquer perfil validado.
Entradas repetidas por nome ou URL equivalente são ignoradas antes do crawl,
da classificação e da exportação; prevalece a primeira ocorrência. O programa
informa quantas entradas foram descartadas. URLs com e sem `www`, com esquemas
HTTP/HTTPS diferentes, barras finais ou parâmetros de consulta são tratadas
como equivalentes quando apontam para o mesmo host e caminho. `--site` também
aceita o nome de uma entrada descartada e seleciona a primeira ocorrência.

```bash
# Coleta pública; não usa APIs pagas.
startup-adherence --mode crawl --sites-file sites.json --profile profiles/digital_twin.json

# Exibe a estimativa total antes de pedir confirmação para Jev e DeepL.
export TYPESAFE_PSN_DIG_TWIN_CLASS='sua-chave'
startup-adherence --mode classify --sites-file sites.json \
  --profile profiles/digital_twin.json --percentage 25

# Recalcula com respostas salvas, sem rede ou chave de API.
startup-adherence --mode score --sites-file sites.json \
  --profile profiles/digital_twin.json
startup-adherence --mode overview --sites-file sites.json \
  --profile profiles/digital_twin.json
startup-adherence --mode export --sites-file sites.json \
  --profile profiles/digital_twin.json --output scores.csv
```

`--mode smoke` coleta até cinco páginas por site e classifica uma amostra
representativa. Também mostra o plano e pede autorização antes de qualquer
chamada paga. `--yes` dispensa a pergunta interativa. `--percentage 0` cancela a
classificação sem chamadas pagas. Tradução é opcional: `--translation auto` ou
`deepl`, com `DEEPL_API_KEY` configurada. O plano informa número de requisições
Jev, requisições DeepL e caracteres estimados, não preços.

Uma coleta recente é reutilizada somente se assinatura de configuração,
manifesto e hash da evidência coincidirem. Use `--force-crawl` para refazer.
`--recover-existing-crawls` recria o estado local a partir de manifesto e
evidência verificáveis, preservando o horário da coleta; não acessa a rede.
Falhas de sites são relatadas separadamente e geram código de saída `1`.
Durante o crawl, o terminal exibe o início de cada site, requisições a robots,
sitemaps e páginas, além da conclusão por site e do avanço total do lote.
Use `-v` para detalhes de respostas e URLs ignoradas; `--no-progress` oculta
as mensagens de progresso.

## Avaliação de texto

```bash
startup-fit --profile profiles/gsd_patient_journey_mapping.json \
  --validate-profile
startup-fit --profile profiles/gsd_patient_journey_mapping.json \
  --input candidate.md --name Candidate --output candidate.fit.json \
  --responses-output candidate.jev.jsonl
startup-fit --profile profiles/gsd_patient_journey_mapping.json \
  --responses candidate.jev.jsonl --name Candidate --output candidate.rescored.json
```

O modo de respostas aceita somente registros com o hash correto das perguntas
e um modelo consistente. Alterar apenas pesos, papéis, limiares ou agregação
permite recalcular localmente; mudar instruções, IDs ou perguntas exige novas
respostas. Arquivos de entrada grandes são divididos em blocos, cada um
tratado como unidade de evidência independente.

No modo de sites, `--mode score` também procura uma execução concluída sob
outra versão do **mesmo ID de perfil** se a nova versão tiver exatamente o
mesmo conjunto de perguntas. A pontuação recalculada fica junto da execução
original, com o hash integral do novo perfil.

## Estrutura e persistência

```text
src/startup_adherence/
  domain/          perfis, identidades, URLs e fórmulas puras
  evidence/        blocos e amostragem
  adapters/        HTTP Jev e DeepL, extração HTML/PDF
  application/     crawl, classificação, replay, relatórios, cache
  storage.py       escrita atômica, logs por execução e ponteiros
  cli.py           orquestração de sites
  text_cli.py      orquestração de textos
profiles/           perfis editáveis, esquema e guias
tests/              contratos de domínio e integração sem rede
```

Cada site guarda `evidence.jsonl`, `manifest.json` e `crawl_state.json` em
`evidence/<site>/`. Cada classificação é isolada em
`profiles/<id>/<version>/runs/<run_id>/`, com `run.json`, respostas completas do
provedor em `responses.jsonl`, traduções em `translations.jsonl` quando usadas
e `result.json`. `current.json` aponta apenas para uma execução completa. Uma
falha mantém o histórico parcial sem substituí-la. Recalcular com outro perfil
de pesos produz `derived/<profile_sha256>.json` ao lado da execução original.
O CSV usa colunas `core__<id>` e `aux__<id>` para evitar colisões.

As páginas e os registros incluem URLs e hashes para rastrear a fonte, mas
o score mede suporte nas evidências coletadas, não a qualidade intrínseca da
empresa. Metadados de cobertura identificam coletas incompletas e amostras.
Resultados antigos em `classification.json`/`jev_chunks.jsonl` não trazem a
identidade verificável das perguntas nem a resposta bruta: mantenha esses
arquivos para auditoria e crie novas execuções quando precisar do contrato
atual. A coleta `evidence.jsonl` pode ser reutilizada para classificar sem
refazer o crawl.

Consulte [o guia de perfis](profiles/HOW_TO_CREATE_PROFILES.md) para desenhar,
validar e versionar novos alvos de avaliação.
