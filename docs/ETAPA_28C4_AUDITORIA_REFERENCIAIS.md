# Etapa 28C.4 — Auditoria de referenciais e sinais (incremental)

**Base auditada:** `AltamiroAJ/python-openplaning`, `feature/mass-properties`, commit `79f944a2` informado pelo usuário. Os módulos `transverse.py`, `lewandowski.py`, `derivatives.py` e `MassProperties` **não foram alterados** neste pacote. A auditoria não constitui validação experimental.

## 1. Origem documental das convenções

- E. M. Lewandowski, *The Dynamics of Marine Craft: Maneuvering and Seakeeping* (2004), cap. 1, p. 1: sistema de corpo destro **x positivo para vante; y para boreste; z para baixo**. Equações do cap. 6 devem ser lidas à luz dessas convenções, salvo redefinições locais.
- E. M. Lewandowski, *The Transverse Dynamic Stability of Planing Craft* (1997), Marine Technology, v. 34, n. 2, p. 109, seção *Equations of Motion*: origem no CG, `x` em direção à proa, `z` para baixo; equações de inércia `K=I_x p_dot - I_zx r_dot`, `N=I_z r_dot-I_zx p_dot`.
- `openplaning/mass_properties.py` (28B): sistema destro **x para vante; y para bombordo; z para cima** e tensor físico `I_xz,tensor = -∫xz dm`.

Não se presume que as convenções de sinais do protótipo `dynroll` correspondam automaticamente às da publicação.

## 2. Transformação espacial correta

O mapeamento de componentes do referencial OpenPlaning FPU (forward/port/up) para o referencial do livro FSD (forward/starboard/down) é a **rotação própria**:

```
R = diag(+1,-1,-1) ; det(R)=+1 ; R.T @ R = I
```

Para vetores polares (posição relativa ao CG, velocidade e forças), momentos axiais e inércia física no CG:

```
v_FSD = R @ v_FPU
F_FSD = R @ F_FPU
M_FSD = R @ M_FPU
I_FSD = R @ I_FPU @ R.T
```

Como a transformação é uma rotação própria, os vetores axiais também transformam por `R`. Uma rotação arbitrária que misture surge, heave ou pitch **não** preserva o subespaço truncado sway–roll–yaw; exige o sistema de 6 graus de liberdade ou redução adicional justificável.

O sinal do produto `I_xz` não pode ser invertido isoladamente. Em particular:

```
I_xz,tensor,FSD = -I_xz,tensor,FPU
P_xz,book,FSD   = integral(x_FSD z_FSD dm)
                 = -I_xz,tensor,FSD
                 = +I_xz,tensor,FPU
```

**Constatação verificável no código 28C.3:** `RigidInertia.Ixz_book_raw` atualmente devolve `-self.Ixz_tensor` embora `self.Ixz_tensor` represente a componente em FPU. Esse valor corresponde ao produto bruto **em FPU**, não ao produto bruto no referencial FSD do livro. Para `I_xz,tensor,FPU=+144 kg m²`, a propriedade antiga devolve `-144 kg m²`, enquanto o produto bruto **FSD** vale `+144 kg m²`.

**Atenção:** essa constatação **não autoriza inverter esse único sinal** na montagem da 28C.3: suas derivadas Y, K, N também precisam ter seus referenciais e projeções reconciliados. Além disso, o termo `Y_r - m U` decorre de convenção cinemática de sway/yaw e deve ser transformado consistentemente. Os resultados de `from_resolved_planing_boat()` permanecem **diagnósticos de implementação**, sem interpretação física validada.

## 3. Transformação completa das equações reduzidas

Considere o descritor:

```
M @ qdot + D @ q + h * phi = 0
q = [v, p, r] ; p = phi_dot
w = [Y, K, N]
```

Na passagem FPU → FSD, com origem mantida no CG:

```
Tq = diag(-1,+1,-1)   # [v,p,r]
Tw = diag(-1,+1,-1)   # [Y,K,N]
s_phi = +1            # phi positivo ao redor do mesmo eixo x
S = diag(-1,+1,+1,-1) # estado [v,phi,p,r]

M_B = Tw @ M_A @ inv(Tq)
D_B = Tw @ D_A @ inv(Tq)
h_B = Tw @ h_A / s_phi
A_B = S @ A_A @ inv(S)
```

Os quatro autovalores e o polinômio característico **normalizado** são invariantes. O polinômio determinantal *não normalizado* pode mudar por um fator global relacionado às transformações de linhas e colunas.

### Sinais de cada derivada, por regra da cadeia

Linhas: `Y,K,N`; colunas: `v,p,r`:

|   | v | p | r | phi |
|---|---|---|---|---|
| Y | + | − | + | − |
| K | − | + | − | + |
| N | + | − | + | − |

A mesma tabela de sinais se aplica às derivadas de aceleração nas colunas `[vdot,pdot,rdot]`. Ela deriva das transformações de equações e coordenadas, **não** de um ajuste para concordar com determinada publicação.

### Escopo de invariância

Os testes fornecidos verificam a mudança física FPU → FSD e escalamentos admitidos com compatibilidade cinemática `p=phi_dot`. Uma mera permutação arbitrária de `(v,p,r)` não constitui teste de referencial físico e pode romper essa compatibilidade.

## 4. Regressões anteriores preservadas

| Evidência do usuário | Regra de auditoria |
|---|---|
| `Ixz=±144 kg m²`: abscissa `−4,162` / `−3,271 s⁻¹` | Trocar apenas `Ixz` é alterar parâmetros, não transformação global; nenhuma invariância exigida |
| `Y_v` do protótipo omite `C_V²` | Preservar variantes etiquetadas, não reescrever coeficientes |
| `N_r` difere entre 1997 e 2004 | Preservar variantes documentais distintas |
| `h1*Y_v/U` de 1997 apresenta incompatibilidade dimensional | Permanecer bloqueada para solver automático |

A regressão de referência de `1600 kg / 12 m/s` deve ser mantida como **baseline do código antes da auditoria**, não alvo de calibração física.

## 5. Testes e critérios de aceite

`tests/test_frame_audit.py` contém 17 testes independentes para:

1. Ortogonalidade e determinante da rotação física.
2. Transformações de forças, momentos, inércia e produto bruto.
3. Conservação dos momentos principais e potência generalizada.
4. Transformação de todas as derivadas por regra da cadeia.
5. Preservação de `phi_dot=p` e rejeição de permutação inadequada.
6. Similaridade entre matrizes `A`, autovalores e polinômios normalizados.
7. Diferença entre mudança de eixos e inversão isolada de `Ixz`.
8. Validação de entradas e rejeição de reflexões como rotações próprias.

**Não se afirma** que esses testes reconciliam as projeções dos dois bordos de 1996, os sinais de todas as derivadas empíricas de 2004 ou os coeficientes de apêndices. São tarefas seguintes antes de emitir qualquer parecer físico de estabilidade.

## 6. Decisão de engenharia

A 28C.4 comprova a transformação espacial e identifica uma ambiguidade concreta na propriedade `Ixz_book_raw` da 28C.3. Para preservar a auditabilidade, **não substitui a função 28C.3**. A correção da cadeia completa (`MassProperties` → derivadas nos eixos FSD → equações do livro) requer um ajuste **atômico** e bateria de regressão específica, sem acomodar valores numéricos publicados.

### Instruções

- Copiar somente `openplaning/stability/frame_audit.py`, `tests/test_frame_audit.py` e este documento para o clone.
- Não sobrescrever `transverse.py`, `derivatives.py`, `lewandowski.py` ou `openplaning.py`.
- Rodar a suíte e realizar commit no branch `feature/mass-properties`.
