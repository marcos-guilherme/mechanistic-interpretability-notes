# Replicação da direção de recusa

Depois de trabalhar com um circuito distribuído no GPT-2 Small, quis testar
uma hipótese mecanística mais compacta: a possibilidade de um comportamento
ser controlado por uma única direção no espaço residual. Nesta replicação,
investiguei essa hipótese no Gemma 2B IT por meio de extração, ablação, adição
e um controle aleatório ortogonal.

A referência é Arditi et al.,
[*Refusal in Language Models Is Mediated by a Single
Direction*](https://arxiv.org/abs/2406.11717). Extraí as direções pela
diferença entre as médias das ativações de pedidos nocivos e inofensivos,
testei cada candidata por ablação e adição e comparei a direção escolhida com
o controle. No Git, mantive métricas agregadas, valores numéricos por exemplo,
configurações e proveniência, sem os textos dos prompts ou das gerações.

Em [`PROCESSO.md`](PROCESSO.md), conto como organizei o estudo, quais decisões
tomei e quais limitações impedem uma conclusão mais ampla.

O protocolo de steering esparso, ainda aguardando execução, está em
[`SPARSE_STEERING.md`](SPARSE_STEERING.md).

## O que observei

Execução de 10 de novembro de 2026 em uma NVIDIA L4:

- direção selecionada: camada 12, último token do template/sufixo, o *newline*
  após o marcador `model`;
- recusa em pedidos nocivos pós-filtro (N=28): 100% na linha de base e 0% após
  ablação;
- recusa indevida em pedidos inofensivos (N=32): 0% na linha de base e 93,75%
  após adição (30/32);
- controle aleatório: 100% de recusa nos pedidos nocivos pós-filtro (N=28);
- KL em pedidos inofensivos (N=32) após ablação: 0,0306, abaixo do limite de
  0,1;
- variação média da entropia cruzada após ablação: -0,4664, contra -0,4652 no
  controle; a proximidade torna essa métrica evidência fraca de especificidade,
  como detalhado no [PROCESSO](PROCESSO.md).

Interpreto esses números como uma reprodução do padrão causal central neste
recorte: remover a direção elimina a recusa, adicioná-la induz recusa e a
ablação com uma orientação aleatória ortogonal não produz o mesmo efeito. A
ablação normaliza a direção antes da projeção, portanto não justifico a
comparação de controle por igualdade de norma. Mantenho a conclusão restrita a
este modelo, seed e recorte; a mesma validação efetiva foi usada para
selecionar a direção e avaliar o efeito final.

## Escopo e desvios

Esta não é uma reprodução integral do artigo. Usei um único modelo, 128
exemplos de treino por classe e amostrei 32 exemplos de validação por classe.
Após o filtro comportamental, o treino efetivo ficou com 121 pedidos nocivos e
127 inofensivos; na validação, foram efetivamente avaliados 28 pedidos nocivos
e 32 inofensivos. A preservação foi medida nos pedidos inofensivos dessa
validação. Não executei todos os modelos, benchmarks de jailbreak ou
conjuntos externos de perda do trabalho original.

O Colab atual usa Python 3.13; por isso a execução remota usa Transformers 4.48.3. Em Python anterior a 3.13, o ambiente mantém Transformers 4.44.2. A lógica de extração, ablação, adição e seleção segue o commit upstream registrado nos artefatos.

## Reproduzir e validar

Os notebooks devem ser executados na ordem:

1. `01_dataset_and_baseline.ipynb`
2. `02_extract_candidate_directions.ipynb`
3. `03_causal_interventions.ipynb`
4. `04_evaluation_and_validation.ipynb`

O Gemma é um modelo com acesso restrito: aceite previamente sua licença na
[página do modelo](https://huggingface.co/google/gemma-2b-it). Prepare o
ambiente e faça login local no Hugging Face dentro do diretório do projeto;
depois, volte à raiz:

```powershell
Set-Location projects/refusal_direction
uv sync
uv run hf auth login
Set-Location ../..
```

Não cole nem exponha o token em notebook, comando, log ou documentação.

Ainda na raiz do repositório, instale a versão verificada da CLI do Colab e
faça a autenticação ADC manual:

```powershell
uv tool install google-colab-cli==0.7.5
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Na raiz do repositório, a sequência operacional do Colab é:

```powershell
python tools/colab/run_refusal.py start --gpu L4
python tools/colab/run_refusal.py sync
python tools/colab/run_refusal.py bootstrap
python tools/colab/run_refusal.py auth-hf
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/01_dataset_and_baseline.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/02_extract_candidate_directions.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/03_causal_interventions.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/04_evaluation_and_validation.ipynb
python tools/colab/run_refusal.py fetch
python tools/colab/run_refusal.py stop
```

Execute `auth-hf` antes do primeiro `execute` e use `fetch` para baixar os
resultados agregados. O comando `auth-hf` transfere para a VM o token já
armazenado pelo cliente Hugging Face local sem imprimi-lo; ao terminar ou se
houver erro, encerre a sessão com `stop`.

Para conferir os resultados baixados:

```powershell
Set-Location projects/refusal_direction
uv run python scripts/validate_results.py
uv run pytest -q
```

As métricas completas estão em `results/final_evaluation.json`.

## Piloto causal em Qwen2.5-Coder-7B (congelado)

Além do recorte Gemma, executei um piloto causal no perfil
`Qwen/Qwen2.5-Coder-7B-Instruct`, revisão fixa
`c03e6d358207e414f1eca0bb1891e29f1db0e242`, em NVIDIA A100 40 GB com
`bfloat16`. A execução `c0818150-f834-4692-b271-4e10a97eb57e` usou o código
registrado em `2a2eb714246174e4f627fcd0061c61143a4e8276`, 128 exemplos de
treino e 32 de validação por classe, e 32 de teste por classe, **sem** filtro
comportamental. O protocolo selecionou direção na validação e mediu condições
finais no teste.

**Operação.** O piloto terminou com sucesso operacional: os estágios rodaram,
os JSONs compactos foram produzidos e a validação offline passou.

**Ciência.** O status final é `causal_gate_failed`. A manipulação foi forte:
recusa nociva no teste caiu de 87,5% para 0% após ablação; a adição levou a
recusa inofensiva a 100%; o controle aleatório manteve 87,5% nos nocivos. A
direção escolhida foi camada 18, offset `-1`. Porém a KL inofensiva após
ablação foi `0,263825`, acima do gate `0,10`. Isso indica sinal causal, mas
selectividade insuficiente neste recorte. O piloto permanece **falhado** no gate;
varreduras de dose em follow-up são exploratórias e **não** reclassificam este
resultado.

Fontes canônicas (somente métricas agregadas e proveniência, sem textos brutos):

- [`results/qwen25_coder_7b/pilot_evaluation.json`](results/qwen25_coder_7b/pilot_evaluation.json)
- [`results/qwen25_coder_7b/final_evaluation.json`](results/qwen25_coder_7b/final_evaluation.json)
- [`results/qwen25_coder_7b/execution_manifest.json`](results/qwen25_coder_7b/execution_manifest.json)

A narrativa completa do congelamento está na seção piloto Qwen em [`PROCESSO.md`](PROCESSO.md).

## Follow-up de dose, J-lens e código no Qwen

Depois da falha do piloto integral, mantive a direção da camada 18 e testei
intensidades menores em splits disjuntos de seleção e confirmação. Esse
follow-up é exploratório e não altera o status `causal_gate_failed` do piloto.

A confirmação escolheu ablação de 20% e adição de 50%. Em 32 pedidos nocivos,
a ablação reduziu a recusa de 96,875% para 43,75%; em 32 pedidos inofensivos,
a adição elevou a recusa de 0% para 31,25%. A KL inofensiva da ablação foi
`0,022759`, e o controle aleatório manteve 96,875% de recusa nociva. Os quatro
gates do follow-up passaram.

Em seguida, ajustei uma J-lens específica com 100 itens do CodeSearchNet e
reservei 20 para holdout. A lente tem 27 matrizes, uma para cada origem entre
as camadas 0 e 26, com alvo na camada 27. No holdout, a J-lens teve
`pass@1/5/10 = 0,12681 / 0,21178 / 0,25167`, contra
`0,11168 / 0,17219 / 0,19351` da logit lens. O corpus não foi deduplicado
contra os benchmarks posteriores, uma limitação registrada nos artefatos.

Por fim, avaliei 32 itens MITRE, 32 MITRE-FRR e 32 HumanEval+ em baseline,
ablação, adição e controle:

- no MITRE, a recusa caiu de 78,125% para 0% com ablação;
- no MITRE-FRR, a adição elevou a recusa de 9,375% para 100%;
- no HumanEval+, a ablação preservou o pass@1 em 40,625%, enquanto a adição o
  reduziu para 21,875% e induziu 65,625% de recusas;
- o controle aleatório permaneceu próximo do baseline e apresentou KL pequena;
- a KL da ablação foi baixa no HumanEval+ (`0,02258`), mas alta no MITRE
  (`4,39113`) e no MITRE-FRR (`1,14368`).

Isso mostra um efeito comportamental útil da ablação parcial nesta amostra,
mas não uma alteração globalmente seletiva. A adição foi indiscriminada e
prejudicou tanto pedidos defensivos quanto tarefas normais de programação.
A revisão humana cega das classificações automáticas de recusa ainda está
pendente.

Fontes compactas:

- [`dose_followup_evaluation.json`](results/qwen25_coder_7b_dose/dose_followup_evaluation.json)
- [`jlens_evaluation.json`](results/qwen25_coder_7b_jlens/jlens_evaluation.json)
- [`04_finalize.json`](results/qwen25_coder_7b_code_eval/04_finalize.json)
- [`execution_manifest.json`](results/qwen25_coder_7b_code_eval/execution_manifest.json)

O método, a leitura da J-lens, as decisões operacionais e as limitações estão
documentados em [`PROCESSO.md`](PROCESSO.md#follow-up-exploratório-no-qwen25-coder-7b).

## J-lens sob intervenção causal

Também medi se as lentes v1 e v2 continuavam recuperando os tokens escolhidos
depois de alterar a direção de recusa. O painel usou os mesmos 32 prompts MITRE
e 32 MITRE-FRR, quatro condições, oito tokens de leitura e as camadas 19, 22,
25 e 26. A execução produziu 256 respostas e 16.384 observações de rank.

Na camada 25, o `pass@5` ficou entre 93,359% e 100%. Sete das oito comparações
principais ficaram dentro da margem de equivalência de cinco pontos
percentuais. A exceção foi a ablação no MITRE: queda de 98,047% para 93,359%,
com intervalo de 95% entre -8,203 e -1,563 pontos. As lentes v1 e v2 tiveram
o mesmo resultado principal, sem vantagem da v2.

A camada 26 permaneceu entre 99,609% e 100%, enquanto as camadas 19 e 22 foram
menos fiéis e mais sensíveis à intervenção. Interpreto o resultado como
evidência de que a leitura tardia permanece útil sob mudança causal, mas não
como robustez uniforme em todas as camadas e condições.

O contexto, o método, os incidentes e os resultados completos estão em
[`CAUSAL_JLENS.md`](CAUSAL_JLENS.md). A fonte numérica canônica é
[`causal_jlens_evaluation.json`](results/qwen25_coder_7b_causal_jlens/causal_jlens_evaluation.json).

## Laboratório visual Qwen (A100)

Depois do follow-up em batch, separei um **laboratório Gradio** para explorar a
mesma direção e J-lens congeladas de forma interativa, sem novo protocolo de
métricas. O perfil Colab `qwen25-coder-lab` sobe o motor, a leitura ao vivo e a
interface com autenticação temporária; `fetch` não se aplica a esse perfil.

Em 16 de novembro de 2026 o smoke em A100 confirmou autenticação, publicação da
URL e geração sincronizada em baseline e ablação 20% com parâmetros fixos
(descritos no caderno de bordo). Exportação pela interface ficou pendente de
smoke remoto; os testes locais cobrem o contrato de exportação explícita.

Guia operacional completo: [`LABORATORIO_QWEN.md`](LABORATORIO_QWEN.md). Contexto
e decisões no [`PROCESSO.md`](PROCESSO.md#laboratório-visual-qwen-novembro-2026).
