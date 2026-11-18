# Laboratório visual de intervenção no Qwen

Documentação operacional do perfil Colab `qwen25-coder-lab`. Descreve como subir
a interface, o que ela faz, o que não faz e como interpretar a J-lens. Não
substitui os JSONs de avaliação em batch nem reescreve as conclusões do piloto
e do follow-up exploratório.

O desenho aprovado em 16 de novembro de 2026 está em
[`docs/superpowers/specs/2026-11-16-qwen-intervention-lab-design.md`](../../docs/superpowers/specs/2026-11-16-qwen-intervention-lab-design.md).

## Objetivo e limites

O laboratório existe para conversar com `Qwen/Qwen2.5-Coder-7B-Instruct` na
revisão fixa do estudo, aplicar baseline, ablação, adição ou controle na
direção congelada e acompanhar, passo a passo, o texto gerado e o top-k da
J-lens numa camada escolhida. A interface escolhe entre J-lens
`v2 (contexto longo)` (padrão) e `v1 (100 amostras)`; o modelo Qwen permanece
fixo. A lente selecionada vale para todos os tokens daquela geração.

É ferramenta de exploração visual, não um novo protocolo de avaliação. Não
executa código gerado, não roda MITRE nem HumanEval+, não compara as quatro
condições em paralelo, não troca de modelo, não ajusta lentes pela interface e
não grava prompts ou respostas no Git nem no pipeline `fetch` dos outros perfis
Qwen.

Uma geração por vez. Encerrar a VM Colab invalida o link e apaga o estado da
sessão no servidor.

## Arquitetura

Três camadas separadas no código:

1. **`InterventionEngine`** (`qwen_lab_runtime.py`): carrega modelo, tokenizer,
   direção e hooks; gera token a token; captura a ativação da camada pedida;
   respeita cancelamento; remove hooks em `finally`.
2. **`LiveJLensReadout` e `LensReadoutRegistry`** (`qwen_lab_readout.py`):
   validam as duas lentes congeladas, recebem a ativação do passo corrente e
   devolvem logits e top-k numéricos da versão escolhida, sem lógica de
   apresentação.
3. **Controller e Gradio** (`qwen_lab_app.py`): valida entradas, orquestra uma
   sessão de geração, faz streaming para os painéis, bloqueia controles durante
   a execução, cuida da exportação explícita e do login da interface.

Cada passo vira um `GenerationEvent` (índice, token emitido, texto acumulado,
condição, camada, top-k, tempos, memória, estado terminal). O Gradio só consome
esses eventos.

Na VM, `colab_start_lab.py` sobe `qwen_run_lab.py` em segundo plano, espera a URL
pública no log e grava um `status.json` operacional sem segredos. O orquestrador
local (`tools/colab/run_refusal.py`) cria credenciais temporárias, envia o JSON
de autenticação uma vez, chama `lab` e imprime URL, usuário e senha só no
terminal local.

## Artefatos fixos

Configuração em `configs/qwen25_coder_7b_lab.json`:

- modelo `Qwen/Qwen2.5-Coder-7B-Instruct`, revisão
  `c03e6d358207e414f1eca0bb1891e29f1db0e242`;
- `bfloat16`, GPU A100, sessão `refusal-qwen-lab`, Gradio `5.49.1`;
- J-lens upstream `anthropics/jacobian-lens` no commit
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`;
- direção copiada de `artifacts/qwen25_coder_7b/selected_direction.pt`, hash
  `0b1cff5e...` (completo no JSON);
- J-lens v1 copiada de
  `artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt`, hash
  `b700a78c90e82f18e057db01cf3dd520beda1f51f1663e2dbf0c1c481efc9e4e`;
- J-lens v2 copiada de
  `artifacts/qwen25_coder_7b_jlens_v2/qwen25_coder_7b_jacobian_lens_v2.pt`, hash
  `b80c084a5ce083d2fc8589a62c9a9b2542b6a9dc6b5c006018cc4956eef37a7a`
  (`source_lens_v2` / `lens_v2_sha256` na config);
- presets: ablação 20%, adição 50%, controle aleatório 20% com seed 1042.

O preflight na subida falha fechado se revisão, dimensões, finitude, CUDA ou
hashes não batem. As duas lentes sobem juntas; só a selecionada participa do
readout. Não há seletor de modelo na interface.

## Como interpretar a J-lens

Os tokens listados no painel da direita são **previsões da J-lens**: os
maiores scores da leitura incremental na camada escolhida para aquele passo.
Indicam o que a lente torna legível na representação observada, não uma
transcrição do raciocínio do modelo nem explicação causal isolada.

A relação causal continua vindo da intervenção na direção de recusa. A lente é
leitura sincronizada durante essa intervenção. Camadas fora do intervalo 0 a 26
são rejeitadas pelo contrato do laboratório.

## Preparação local

O checkpoint do Qwen usado aqui é público. Ainda assim, faça login no Hugging
Face para evitar limites anônimos durante o download, sem expor o token em
notebook ou documentação:

```powershell
Set-Location projects/refusal_direction
uv sync
uv run hf auth login
Set-Location ../..
```

Na raiz do repositório, instale a CLI Colab verificada e autentique ADC:

```powershell
uv tool install google-colab-cli==0.7.5
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Confirme que os três `.pt` locais existem (direção, J-lens v1 e J-lens v2) e
que os SHA-256 coincidem com `configs/qwen25_coder_7b_lab.json`. O `sync` do
perfil lab limpa partes remotas antigas, envia a direção inteira e as duas
lentes em partes de 16MiB com manifesto SHA-256 (`lab_source_lens.pt` e
`lab_source_lens_v2.pt`) e falha se tamanho, contagem ou hash não conferirem.
O instalador remoto remonta as duas lentes e as instala atomicamente nos
caminhos congelados. Sobrescreva o caminho da v2 com `--lens-v2-path` em
`qwen_run_lab.py` se precisar apontar para um artefato já extraído. O perfil
code-eval continua a usar partes de 64MiB.

## Comandos do perfil lab

Todos a partir da raiz do repositório. O perfil `qwen25-coder-lab` força A100.
A sessão dedicada é `refusal-qwen-lab`, separada dos notebooks Gemma/Qwen em
batch.

```powershell
python tools/colab/run_refusal.py start --profile qwen25-coder-lab
python tools/colab/run_refusal.py sync --profile qwen25-coder-lab
python tools/colab/run_refusal.py bootstrap --profile qwen25-coder-lab
python tools/colab/run_refusal.py auth-hf --profile qwen25-coder-lab
python tools/colab/run_refusal.py lab --profile qwen25-coder-lab
python tools/colab/run_refusal.py status --profile qwen25-coder-lab
python tools/colab/run_refusal.py stop --profile qwen25-coder-lab
```

Ordem obrigatória: `auth-hf` antes de depender do modelo na VM; `lab` só depois
de `bootstrap` ter instalado dependências e extraído o pacote. `stop` é
obrigatório ao terminar, mesmo se `lab` falhar depois que a VM subiu.

### O que cada comando faz

- **start**: abre VM A100 com nome de sessão fixo do perfil.
- **sync**: envia código do projeto e os três artefatos congelados (sem datasets).
- **bootstrap**: roda o instalador remoto, fixa Gradio 5.49.1 e instala J-lens
  com `--no-deps` de propósito para manter o Transformers do `pyproject.toml`;
  remonta as lentes v1 e v2 com manifesto e valida hashes declarados na config.
- **auth-hf**: transfere o token HF já salvo localmente, sem imprimi-lo.
- **lab**: gera usuário e senha locais, envia arquivo de auth de uso único,
  inicia o servidor Gradio em background e imprime URL pública, credenciais e
  aviso do relay.
- **status**: lê apenas estado operacional remoto via `qwen_run_lab_status.py`;
  não baixa resultados nem aceita payloads com senha ou prompt.
- **stop**: encerra a sessão Colab e limpa staging local do perfil lab.

## J-lens e Transformers

O instalador remoto usa `pip install --no-deps` no Jacobian-lens de propósito:
as versões de Transformers continuam as do `pyproject.toml` (`4.44.2` ou
`4.48.3`, conforme o Python). Depois da instalação, o runtime confere a versão
efetiva contra esses pins.

O identificador `runtime_compatibility_smoke` =
`2026-11-16-qwen7b-hooks-jlens-transport-unembed` registra o smoke de 16 de
novembro de 2026: Qwen 7B com hooks e transporte/unembed da J-lens no ambiente
pinado. Não é uma declaração de compatibilidade geral nem um convite a atualizar
Transformers.

## Relay externo

O link público do Gradio passa por um **relay externo** (domínios `gradio.live`
ou `gradio.space`). Prompts e respostas da interface transitam por esse serviço
enquanto a sessão estiver aberta. Trate o laboratório como canal não confiável
para conteúdo sensível.

## Credenciais temporárias

Cada `lab` gera par usuário/senha com `secrets.token_urlsafe`. A senha aparece
só no terminal local da pessoa que rodou o comando; não entra em `status.json`,
logs versionados nem exportações padrão. O arquivo local de auth em
`.colab-staging` é apagado em `finally`; na VM, `qwen_run_lab.py` consome e
remove o JSON remoto antes de abrir o Gradio.

## Não usar `fetch`

`python tools/colab/run_refusal.py fetch --profile qwen25-coder-lab` é recusado
de propósito. O laboratório não produz JSONs para o pipeline agregado dos
outros perfis. Métricas científicas continuam nos diretórios `results/` das
execuções em batch. Exportação de uma conversa é só pela ação **Exportar sessão**
na interface, para o disco da pessoa usuária.

## Privacidade e exportação explícita

Por padrão no servidor:

- prompts e gerações ficam em memória;
- não há histórico persistente no Colab além do necessário à execução corrente;
- logs operacionais evitam texto livre de prompt ou resposta;
- pesos, lente, token HF e senha não são expostos na página.

**Exportar sessão** grava um JSON sob demanda (parâmetros, tokens, scores,
tempos), sem pesos nem credenciais. Esse arquivo não é puxado automaticamente
para o repositório.

## Repetir o smoke de hardware

Smoke mínimo acordado no desenho:

1. Subir a cadeia start → sync → bootstrap → auth-hf → lab.
2. Abrir a URL com as credenciais impressas localmente.
3. Usar um prompt inofensivo e curto de programação (não MITRE, não executar
   código gerado).
4. Rodar **baseline** e **ablação 20%** com temperatura 0, camada J-lens 18,
   top-k 5, limite de 6 novos tokens. A J-lens padrão é `v2 (contexto longo)`;
   compare com `v1 (100 amostras)` numa geração seguinte, sem trocar de lente
   no meio do texto.
5. Confirmar que ambos atingem o passo 6 com estado `max_tokens` (ou
   equivalente documentado na UI) e que token emitido e top-k da lente
   atualizam em sincronia a cada passo.
6. Opcional: testar exportação na interface e verificar que o JSON não traz
   segredos (coberto localmente pelos testes; ver lacuna abaixo).
7. `stop` e confirmar que não resta sessão ativa.

Registro honesto de 16 de novembro de 2026 em A100-SXM4-40GB: autenticação
temporária funcionou; baseline e ablação 20% passaram nos critérios acima; não
persisto aqui o texto do prompt nem trechos de geração. Na primeira subida houve
atraso porque o log do servidor estava bufferizado; o commit `71788f6` passou a
lançar o Python com `-u`, e a segunda subida publicou a URL em cerca de 40
segundos. O teste remoto de exportação não terminou porque a sessão Colab
encerrou antes da ação na interface: exportação explícita está coberta pelos
testes locais, mas a transferência real do JSON pelo relay ainda merece um
smoke futuro. Ao final, `colab sessions` listou nenhuma sessão ativa.

## Verificar o encerramento

Depois de `stop`:

```powershell
colab sessions
```

A lista deve ficar vazia ou sem `refusal-qwen-lab`. Se `lab` falhou após
`start`, rode `stop` mesmo assim para evitar custo e arquivo de auth remoto
órfão.

O perfil limpa staging local em `.colab-staging` ligado ao lab ao parar.

## Solução de problemas

### Startup

- **Preflight falhou na VM**: confira revisão do modelo, presença da A100,
  hashes locais dos `.pt` e se `auth-hf` rodou antes do download.
- **`lab` estourou tempo esperando URL**: primeira carga do Qwen 7B mais lente
  pode atrasar; leia `status` depois (o servidor pode ainda estar subindo). O
  instalador fixa espera longa no remoto; em dia lento, repita `status` antes
  de subir de novo.
- **Log sem URL na primeira tentativa (buffer)**: versões anteriores ao `71788f6`
  podiam segurar stdout; a linha de comando remota usa `python -u` no lançador.
  Se a URL só aparecer no log remoto, sincronize código com esse commit.
- **Servidor morreu cedo**: se o upload de auth na VM funcionou e o processo
  caiu, o JSON remoto pode existir até `stop`. Não imprima esse ficheiro para
  depurar; destrua a sessão.

### Status

- **`status` recusa payload**: por contrato, não aceita campos de senha ou
  prompt. Se o ficheiro remoto foi corrompido manualmente, recrie com `lab`.
- **URL em `status` mas interface inacessível**: relay ou sessão expirada;
  confirme `colab sessions` e repita a cadeia se necessário.

### OOM

Durante geração, falta de VRAM interrompe o passo, remove hooks e devolve erro
seguro na UI. Não há fallback silencioso de dtype ou GPU. Reduza `max_new_tokens`,
feche outras cargas na VM (este perfil assume A100 dedicada) e tente de novo.
Se o OOM ocorrer na carga inicial, confirme que nenhum outro processo grande
divide a mesma VM.

## Custos e encerramento

A100 no Colab é recurso pago ou com cota limitada. Mantenha a janela do
laboratório só enquanto precisar explorar. Cada minuto com a VM ligada conta,
mesmo sem gerar texto.

Sempre:

```powershell
python tools/colab/run_refusal.py stop --profile qwen25-coder-lab
colab sessions
```

Isso derruba o relay, invalida credenciais daquela execução e evita deixar token
HF ou lente expostos num disco efémero esquecido.

## Testes locais antes de subir

No diretório do projeto:

```powershell
uv run --with gradio==5.49.1 pytest tests/test_qwen_lab.py tests/test_qwen_lab_readout.py tests/test_qwen_lab_runtime.py tests/test_qwen_lab_app.py tests/test_qwen_lab_transport.py tests/test_colab_orchestrator.py -q
```

Esses testes cobrem contratos do motor, paridade incremental da lente, recusa de
`fetch`, consumo do arquivo de auth, ausência de senha em logs e exportação
explícita sem vazamento de metadados proibidos.