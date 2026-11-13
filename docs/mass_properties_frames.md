# Etapa 28B.3 — Transformações de sistemas de coordenadas

## Referencial e unidades

O módulo `MassProperties` define por padrão um referencial **destro** ligado ao corpo:
`x` positivo para vante desde o espelho de popa; `y` para bombordo; `z`
para cima desde o datum do fundo/quilhas. Unidades: m, kg, kg·m².

`RigidTransform(rotation=R, translation=t)` converte do referencial de origem
**source** para o de destino **target**:

\[\mathbf r_t=\mathbf R\mathbf r_s+\mathbf t\]

`translation` representa a posição **da origem source descrita no target**.
Portanto a aplicação de uma translação não altera a inércia sobre o CG,
enquanto a inércia em relação a outro ponto deve usar o teorema dos eixos paralelos.

\[\mathbf I_{G,t}=\mathbf R\mathbf I_{G,s}\mathbf R^T\]

As matrizes são **tensores físicos de inércia de massa**, com diagonais
`∫(y²+z²)dm`, etc., e fora da diagonal `−∫xy dm`, `−∫xz dm`, `−∫yz dm`.
Essas entradas **não** são os produtos crus `∫xy dm` utilizados por alguns
textos; confira sempre a convenção antes de transcrever `I_xz`.

## Exemplos

```python
import numpy as np
from openplaning.frames import (
    RigidTransform, FORWARD_PORT_UP_TO_FORWARD_STARBOARD_DOWN,
)
from openplaning.mass_properties import MassProperties

masses = MassProperties()
masses.add_box(1200, [5, 0, 0.8], [2, 1.5, 0.6])
masses.add_point_mass(250, [3, 0.4, 1.2])
boat_frame = masses.calculate()

# Rotação de 90° em torno do eixo z; origem deslocada no referencial de destino.
T = RigidTransform.from_axis_angle([0, 0, 1], np.pi/2,
                                    translation=[10, -2, 1])
transformed = boat_frame.transformed(T)
assert np.allclose(transformed.inertia_tensor,
                   T.rotation @ boat_frame.inertia_tensor @ T.rotation.T)

# Conversão de x-vante/y-bombordo/z-cima para x-vante/y-boreste/z-baixo.
# NÃO presume que Lewandowski usa esse referencial.
other = masses.transformed(FORWARD_PORT_UP_TO_FORWARD_STARBOARD_DOWN)
```

`MassProperties.transformed(T)` cria uma **nova** coleção, com todos os
componentes expressos no target, sem modificar a coleção original.
`MassResult.transformed(T)` transforma diretamente o resultado agregado.
`T.inverse()` produz a transformação inversa e `T2.compose(T1)` significa
aplicar T1 e depois T2.

## Cuidados relativos à futura implementação de Lewandowski

1. Conferir documentalmente orientação de eixos, sinal dos momentos e
   convenção *produto de inércia* em Lewandowski (1997/2004), separadamente
   da escolha do referencial do OpenPlaning.
2. `Ixx`, `Iyy`, `Izz` são momentos de inércia de massa de corpo rígido, não
   adicionados hidrodinamicamente. `Ixz` no tensor é uma entrada fora da
   diagonal **negativa** do produto cru de coordenadas, por definição.
3. Para uma matriz de equações de movimento que use definição oposta de
   produto de inércia, converter antes da montagem, nunca simplesmente copiar
   a entrada `[0,2]`.
4. O CG e os momentos devem ser transformados conjuntamente; a transformação
   de uma única coordenada sem o tensor inteiro é incorreta em geral.
5. O campo `r_g` original de `PlaningBoat` pressupõe a direção de pitch
   do referencial de análise. Seu valor precisa ser recalculado depois de uma
   rotação arbitrária de eixos; a conversão específica acima preserva a direção
   do eixo transversal até o sinal, logo preserva o valor de `Iyy`.
6. Os métodos continuam independentes de `get_steady_trim()` e dos modelos
   hidrodinâmicos existentes.

## Testes

Executar na raiz do repositório:

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Limites

- A biblioteca não foi alterada para incorporar o objeto `MassProperties`
  diretamente em `PlaningBoat` (reservado para 28B.4).
- Não há um **referencial Lewandowski** definido automaticamente sem comprovar
  os eixos da fonte primária.
- Tensores de inércia e transformações têm convenção SI; nenhum ensaio
  experimental de embarcação está sendo reivindicado nesta entrega.
