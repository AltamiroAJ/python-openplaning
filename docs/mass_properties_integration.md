# 28B.4 — Integração controlada da massa rígida a PlaningBoat

O adaptador `openplaning.mass_integration` associa propriedades de massa a um
`PlaningBoat` **sem alterar os campos existentes**. O método `set_mass_properties`
rejeita divergências de peso, LCG, VCG e raio de giração. Também rejeita CG
transversal não nulo, pois o equilíbrio atual presume simetria de casco/carga.

## Referencial

A referência padrão é um referencial destro de corpo rígido, com x para vante
a partir do espelho, y para bombordo, z para cima, e origem na linha de quilha.
Essa convenção deve corresponder aos campos `lcg` e `vcg` de `PlaningBoat`.
Caso a fonte dos componentes esteja em outra orientação/origem, forneça
`frame_transform=RigidTransform(...)` (origem/rotação da fonte convertidas
para a referência da embarcação). Não se presume automaticamente convenção
Naval Architects, NED ou a convenção das equações de Lewandowski.

A matriz de inércia representa somente **massa rígida**, em kg m²:

    I_xz,tensor = - integral(x*z dm)
    P_xz,raw    = + integral(x*z dm) = -I_xz,tensor

A passagem para as matrizes de Lewandowski exige conferência posterior da
convenção dos produtos de inércia. O sistema não incorpora massa adicionada.

## Exemplo da interface após aplicar a pequena alteração em PlaningBoat

```python
from openplaning import PlaningBoat
from openplaning.mass_properties import MassProperties

mass = MassProperties()
mass.add_box(1200, center=(5.0, 0.0, 1.2), dimensions=(2, 3, 4))

# Exemplo: criar PlaningBoat com weight=1200*g, lcg=5.0,
# vcg=1.2, r_g=sqrt(Iyy/1200), mais parâmetros operacionais existentes.
boat.set_mass_properties(mass)  # falha com ValueError se não houver coerência
rigid = boat.get_mass_properties_snapshot()
print(rigid.inertia_terms)  # Ixx, Iyy, Izz, Ixz_tensor, Pxz_raw, ...
```

## Sem sincronização implícita

A presente etapa não oferece `update_inputs=True`: seria necessário invalidar
resultados já calculados pelo `PlaningBoat` e reconstruir forças, massa e estado
operacional de forma transacional. É mais seguro criar o objeto `PlaningBoat`
com os valores derivados de `mass.calculate()` e vincular o modelo apenas após
verificar a consistência.

Para inspecionar divergências sem qualquer modificação, use:

```python
from openplaning.mass_integration import check_mass_consistency
report = check_mass_consistency(boat, mass)
print(report.consistent, report.differences, report.lateral_cg)
```

`get_mass_properties_snapshot()` retorna uma cópia defensiva do objeto
congelado de massa/inércia; mutações posteriores em `MassProperties` não
mudam implicitamente as inércias associadas.

## Limites

- Não há alteração no solver de Savitsky nem em `get_steady_trim()`.
- Não há reequilíbrio para condição de CG assimétrico, tanques parcialmente
  cheios, massas variáveis ou superfície livre.
- `MassProperties` pode estimar distribuição estrutural, não inferi-la da
  geometria hidrodinâmica.
- A etapa 28C deverá reconciliar eixos, unidades e sinais de Lewandowski.
