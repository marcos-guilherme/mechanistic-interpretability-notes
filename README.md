# Replicações de interpretabilidade mecanística

Criei este repositório para armazenar e documentar meu processo prático de
entrada na área de interpretabilidade mecanística. Mais do que reunir código,
quero registrar o caminho que percorri para transformar artigos em
experimentos executáveis: o que tentei reproduzir, as decisões que tomei, os
problemas que encontrei e até onde cada resultado realmente permite concluir.

Organizei o trabalho em três estudos. Comecei pelo circuito de identificação
do objeto indireto em GPT-2 Small, avancei para uma direção causal de recusa no
Gemma 2B IT e, por fim, preparei uma intervenção conceitual em J-space no
Llama 3.1 8B Instruct. Nem todas as etapas terminaram da mesma forma: duas
produziram resultados científicos e a tentativa com J-space ficou inconclusiva
antes da inferência, por falta de autorização para acessar os pesos do modelo.
Mantive esse resultado porque a falha operacional também faz parte de um
processo experimental reproduzível.

## Como este repositório está organizado

Separei cada estudo em seu próprio projeto, com ambiente Python, notebooks,
testes e resultados. A documentação compartilhada conecta esses experimentos:

- o [índice das replicações](docs/replicacoes/README.md) funciona como mapa do
  trabalho;
- a [metodologia comum](docs/replicacoes/METODOLOGIA.md) reúne os critérios que
  passei a aplicar para planejar, executar e avaliar cada reprodução;
- o [modelo de replicação](docs/replicacoes/MODELO.md) transforma esses
  critérios em um roteiro reutilizável;
- as [fontes e referências](docs/REFERENCIAS.md) distinguem os artigos
  originais, os materiais de formação e as implementações que orientaram os
  estudos.

## Estudos

### [Circuito IOI em GPT-2 Small](projects/ioi)

Usei a tarefa de identificação do objeto indireto como primeiro contato com a
análise de circuitos. O projeto combina uma etapa didática, na qual reconstruí
as ferramentas passo a passo, e uma etapa mais próxima do artigo, com
resultados locais para baseline, caminhos causais, *Backup Name Movers*,
*faithfulness*, *completeness*, *minimality* e protocolos adversariais.

### [Direção de recusa no Gemma 2B IT](projects/refusal_direction)

Depois do circuito distribuído de IOI, investiguei uma hipótese mais compacta:
se a recusa poderia ser controlada por uma única direção no espaço residual.
A execução em uma NVIDIA L4 produziu resultados para ablação, adição, controle
aleatório e preservação. A conclusão é causal, mas permanece restrita ao
modelo, à seed e ao recorte avaliados.

### [J-space no Llama 3.1 8B Instruct](projects/jspace_llama)

Neste terceiro estudo, preparei uma intervenção sobre um conceito
intermediário, usando uma J-lens pré-ajustada e a troca `spider → ant`. A
execução completa em uma NVIDIA L4 carregou o modelo e a lente, realizou o
readout e aplicou a intervenção e os controles. O protocolo, porém, permaneceu
inconclusivo: o baseline não produziu uma resposta numérica válida para o
parser, o readout não encontrou `spider` e a intervenção-alvo vazou `ant` e
gerou repetição, sem produzir seis de forma válida.

## Como ler os resultados

Os READMEs e arquivos `PROCESSO.md` contam o raciocínio, as escolhas e os
limites de cada estudo. Os JSONs em `results/` são as fontes canônicas dos
números e da proveniência. Procurei manter essas duas camadas separadas para
que uma narrativa mais acessível não substitua a evidência que a sustenta.

Os comandos devem ser executados a partir do diretório de cada projeto, exceto
quando apontarem explicitamente para `tools/`. Pesos, caches, credenciais,
ativações e outros artefatos pesados ou sensíveis permanecem fora do Git.
