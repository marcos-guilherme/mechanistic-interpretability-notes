# J-lens v2 no Qwen: execução longa no Colab

Documentação operacional do perfil `qwen25-coder-jlens-v2`. Descreve corpus,
probe, seleção do ajuste, checkpoints, recuperação, holdout, comparação com a
v1 e limites explícitos. Não substitui os JSONs do ajuste v1 nem reescreve as
conclusões já registradas em `PROCESSO.md` para a lente curta.

Perfil JSON: `configs/qwen25_coder_7b_jlens_v2.json`. Operador local:
`tools/colab/run_refusal.py`.

## Objetivo e o que muda em relação à v1

A v1 ajustou 100 funções Python com contexto curto e holdout de 20 exemplos. A
v2 aumenta o contrato para **512 tokens** por exemplo, fixa **200 amostras** no
holdout e deixa o tamanho do ajuste (`fit_count`) ser escolhido por probe entre
candidatos até 800, com projeção limitada a **12 horas** de GPU. Os artefatos
ficam em `artifacts/qwen25_coder_7b_jlens_v2/` e `results/qwen25_coder_7b_jlens_v2/`.

O laboratório visual (`qwen25-coder-lab`) e a avaliação de código em batch
continuam usando a lente v1 congelada em
`artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt`. A v2 é um
experimento paralelo de ajuste e de comparação no holdout ampliado, não uma
troca automática da lente usada nas intervenções já documentadas.

## Corpus e holdout

- Repositório: `code-search-net/code_search_net`, split `train`, config `python`,
  revisão fixa no JSON.
- Cada exemplo precisa atingir pelo menos 512 tokens antes do truncamento no
  `skip_first` (16).
- O holdout tem exatamente 200 itens, fixos após a preparação; o conjunto de
  ajuste é um prefixo do pool restante, recortado pelo `fit_count` selecionado.
- Lista de exclusão em `exclusion_hashes_path` reduz vazamento contra benchmarks
  usados em outras etapas do estudo.

## Probe e seleção do `fit_count`

Antes do ajuste longo, o pipeline mede tempo por amostra com `dim_batch` da
ordem de probe (32, 16, 8, 4, 2, 1) e escolhe o maior candidato em
`fit_count_candidates` cuja projeção total caiba em `max_projected_hours` (12).

Na primeira execução real, em 17 de novembro de 2026, os lotes 32, 16 e 8
esgotaram a memória de uma A100 de 80 GB antes do ajuste. Por isso, a ordem foi
estendida até 1. A mudança reduz o pico de memória sem alterar os exemplos nem
o cálculo; o custo é processar as dimensões da Jacobiana em mais etapas.

## Resultado do probe na A100

A execução real mostrou que o desenho de contexto integral de 512 tokens não
cabe no orçamento atual:

- a primeira tentativa encerrou corretamente após `dim_batch` 32, 16 e 8
  esgotarem os 80 GB da A100;
- a ordem foi ampliada para 4, 2 e 1 e o probe foi reiniciado;
- após 61 minutos, nenhuma tentativa adicional havia produzido uma Jacobiana
  completa nem um checkpoint;
- a sessão foi encerrada manualmente, e o servidor confirmou que nenhuma GPU
  permaneceu ativa.

Nenhuma lente v2 foi ajustada nessa execução. O resultado é um diagnóstico de
inviabilidade do cálculo integral de 512 tokens, não um resultado negativo
sobre a qualidade da J-lens.

O próximo desenho a avaliar é percorrer cada função em janelas de 128 tokens.
Isso cobre os 512 tokens de cada exemplo em quatro observações que cabem em
memória. Essa alternativa muda a interpretação: ela aumenta a cobertura de
tokens, mas não mede uma Jacobiana com 512 tokens simultaneamente no contexto.
Essa diferença deve permanecer explícita na comparação com a lente original.

O valor escolhido fica gravado no checkpoint remoto (`fit_count`, `n_done`,
`next_idx`, somas de Jacobiano por camada fonte). A comparação v2 versus v1 no
holdout exige os mesmos `example_id` e a lente v1 carregada do caminho
`source_v1_lens` do JSON.

## Sequência de comandos (operador local)

Executar na raiz do repositório, com credenciais Colab e Hugging Face já
configuradas:

```powershell
python tools/colab/run_refusal.py start --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py sync --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py bootstrap --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py auth-hf --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py jlens --profile qwen25-coder-jlens-v2 --timeout 46800
python tools/colab/run_refusal.py checkpoint-backup --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py fetch --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py stop --profile qwen25-coder-jlens-v2
```

`jlens` dispara o notebook de ajuste e validação na A100. `fetch` traz JSONs
agregados e o tensor final quando o remoto tiver terminado. `stop` encerra a
VM depois de tentar um último backup.

## Checkpoints e recuperação

Durante o ajuste, cada amostra concluída atualiza um **checkpoint** remoto em
`artifacts/qwen25_coder_7b_jlens_v2/` (tensores parciais, progresso e metadados
de identidade da corrida).

Para cópia local enquanto o kernel ainda está ocupado, use
`checkpoint-backup`: transporte pela API de arquivos do Colab, com progresso
impresso no terminal. Pode rodar **durante** o fit. O arquivo costuma ficar em
torno de 1,4 GB; o backup pode levar vários minutos e consumir memória
substancial na máquina local.

`status` lê apenas `status.json` e eventos leves no remoto. O Colab executa um
kernel por sessão; com o fit ativo, uma chamada `status` pode **enfileirar** até
o kernel liberar. Para acompanhar progresso sem bloquear o ajuste, prefira
`checkpoint-backup` ou observe os eventos já sincronizados após um backup.

Se a sessão cair, recupere com novo `start`, `sync`, `bootstrap` e relance
`jlens`: o remoto deve retomar a partir do último checkpoint válido se os
artefatos persistirem no volume da VM.

## Timeout, parada e alarmes

O argumento `--timeout 46800` no `jlens` é o limite de **inatividade de saída**
do `colab exec`, em segundos (13 horas), não um cronômetro rígido de parede. Um
fit longo pode continuar enquanto houver logs periódicos.

`stop` tenta um backup final por até 600 segundos e depois prioriza desligar a
A100. Se o alarme de parada indicar que a A100 pode ainda estar ativa, repita
imediatamente o mesmo comando `stop` com o mesmo perfil.

## Interpretação dos resultados

As métricas de holdout (ranks da J-lens v2, da v1 congelada e da logit lens)
medem **fidelidade descritiva da lente** no corpus de código, não correção de
programas nem taxas de recusa comportamental por si só. Elas alimentam a
comparação planejada no JSON de avaliação v2; não invalidam nem confirmam, só
com os números do piloto causal, do follow-up de dose ou do code-eval MITRE.

## Limites explícitos

- Ela não substitui a lente v1 no laboratório nem no code-eval até decisão
  explícita documentada fora deste perfil.
- Não garante ausência de vazamento entre ajuste e benchmarks externos; a lista
  de exclusão mitiga, mas não cobre todos os recortes do estudo.
- Checkpoints grandes e backups pela API de arquivos são frágeis a quedas de
  rede e a limites de memória local.
- A projeção de 12 horas depende do probe no hardware atual; mudança de GPU ou
  de revisão do modelo invalida a extrapolação.