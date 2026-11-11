----------------------------- MODULE BeamSearch -----------------------------
EXTENDS Naturals, Integers, Sequences, FiniteSets, TLC
CONSTANTS Focused, Mutation
VARIABLES phase, neuron, rawConcepts, concepts, width, cap, iteration, beam, expressions,
          eligible, generated, unique, representatives, selected, ancestry, previous
vars == <<phase, neuron, rawConcepts, concepts, width, cap, iteration, beam, expressions,
          eligible, generated, unique, representatives, selected, ancestry, previous>>
Min(S) == CHOOSE x \in S : \A y \in S : x <= y
Max(a,b) == IF a > b THEN a ELSE b
Bits(m) == {i \in 1..2 : (m \div (2^(i-1))) % 2 = 1}
Bit(m,i) == (m \div (2^(i-1))) % 2
Apply(m,n,o) ==
  LET B(i) == IF o=1 THEN Bit(m,i)*Bit(n,i)
              ELSE IF o=2 THEN Max(Bit(m,i),Bit(n,i))
              ELSE Bit(m,i)*(1-Bit(n,i))
  IN B(1)+2*B(2)
Eval(p,c,o) == CASE o=1 -> p \intersect c [] o=2 -> p \union c [] OTHER -> p \ c
Numer(m) == Cardinality(Bits(m) \intersect Bits(neuron))
Denom(m) == Max(1,Cardinality(Bits(m) \union Bits(neuron)))
GT(a,b) == Numer(a)*Denom(b) > Numer(b)*Denom(a)
EQ(a,b) == Numer(a)*Denom(b) = Numer(b)*Denom(a)
Take(s,k) == SubSeq(s,1,IF Len(s)<k THEN Len(s) ELSE k)
Ascending(S) == [k \in 1..Cardinality(S) |-> CHOOSE i \in S : Cardinality({j \in S:j<i})=k-1]
InitialIndices == {i \in DOMAIN rawConcepts : Numer(rawConcepts[i])>0}
InitialOrder == [k \in 1..Cardinality(InitialIndices) |->
  CHOOSE i \in InitialIndices : Cardinality({j \in InitialIndices :
    GT(rawConcepts[j],rawConcepts[i]) \/ (EQ(rawConcepts[j],rawConcepts[i]) /\ j<i)})=k-1]
Decoded(r) == [parent |-> eligible[((r-1) \div (Len(concepts)*3))+1],
                concept |-> (((r-1) \div 3) % Len(concepts))+1,
                operation |-> ((r-1)%3)+1]
RowsFor(m) == {i \in DOMAIN generated:generated[i]=m}
First(m) == Min(RowsFor(m))
GenerationOrder == [k \in 1..Len(unique) |-> CHOOSE i \in DOMAIN unique :
  Cardinality({j \in DOMAIN unique:representatives[j]<representatives[i]})=k-1]
ScoreOrder == [k \in 1..Len(unique) |-> CHOOSE i \in DOMAIN GenerationOrder :
  Cardinality({j \in DOMAIN GenerationOrder : GT(unique[GenerationOrder[j]],unique[GenerationOrder[i]]) \/
    (EQ(unique[GenerationOrder[j]],unique[GenerationOrder[i]]) /\ j<i)})=k-1]
UniqueOrder == [k \in DOMAIN ScoreOrder |-> GenerationOrder[ScoreOrder[k]]]
Init == /\ phase="initialize" /\ neuron \in 0..3
        /\ rawConcepts \in (IF Focused THEN {<<1,2>>,<<2,1>>} ELSE [1..2 -> 0..3])
        /\ concepts=[k \in DOMAIN InitialOrder |-> rawConcepts[InitialOrder[k]]]
        /\ width \in (IF Focused THEN {2} ELSE 1..2)
        /\ cap \in (IF Focused THEN {2} ELSE 0..2)
        /\ iteration=0 /\ beam= << >> /\ expressions= << >> /\ eligible= << >>
        /\ generated= << >> /\ unique= << >> /\ representatives= << >>
        /\ selected= << >> /\ ancestry= << >> /\ previous= << >>
Initialize == /\ phase="initialize"
  /\ LET ids == Take(concepts,width) IN
     /\ beam'=[k \in DOMAIN ids |-> ids[k]]
     /\ expressions'=[k \in DOMAIN ids |-> Bits(ids[k])]
  /\ phase'="ready"
  /\ UNCHANGED <<neuron,rawConcepts,concepts,width,cap,iteration,eligible,generated,unique,representatives,selected,ancestry,previous>>
ChooseEligible == /\ phase="ready" /\ Len(beam)>0 /\ iteration<cap
  /\ \E E \in SUBSET DOMAIN beam :
       /\ (iteration=0 => E \in {{},DOMAIN beam})
       /\ eligible'=Ascending(E)
  /\ phase'="eligible"
  /\ UNCHANGED <<neuron,rawConcepts,concepts,width,cap,iteration,beam,expressions,generated,unique,representatives,selected,ancestry,previous>>
Generate == /\ phase="eligible" /\ Len(eligible)>0
  /\ generated'=[r \in 1..(Len(eligible)*Len(concepts)*3) |->
     LET d==Decoded(r) IN Apply(beam[d.parent],concepts[d.concept],d.operation)]
  /\ phase'="generated"
  /\ UNCHANGED <<neuron,rawConcepts,concepts,width,cap,iteration,beam,expressions,eligible,unique,representatives,selected,ancestry,previous>>
Deduplicate == /\ phase="generated"
  /\ LET u==Ascending({generated[i]:i \in DOMAIN generated}) IN
     /\ unique'=u
     /\ representatives'=[k \in DOMAIN u |-> IF Mutation="last" THEN
           CHOOSE x \in RowsFor(u[k]): \A y \in RowsFor(u[k]):x>=y
           ELSE First(u[k])]
  /\ phase'="deduplicated"
  /\ UNCHANGED <<neuron,rawConcepts,concepts,width,cap,iteration,beam,expressions,eligible,generated,selected,ancestry,previous>>
TieOrder == [k \in 1..Len(unique) |-> CHOOSE i \in DOMAIN unique :
  Cardinality({j \in DOMAIN unique : GT(unique[j],unique[i]) \/
    (EQ(unique[j],unique[i]) /\ j<i)})=k-1]
Select == /\ phase="deduplicated"
  /\ selected'=Take(IF Mutation="tie" THEN TieOrder ELSE UniqueOrder,width)
  /\ phase'="selected"
  /\ UNCHANGED <<neuron,rawConcepts,concepts,width,cap,iteration,beam,expressions,eligible,generated,unique,representatives,ancestry,previous>>
Commit == /\ phase="selected"
  /\ LET ids==[k \in DOMAIN selected |-> representatives[selected[k]]]
         records==[k \in DOMAIN ids |-> Decoded(ids[k])]
         parents==[k \in DOMAIN ids |-> IF Mutation="remap" THEN
              ((ids[k]-1) \div (Len(concepts)*3))+1 ELSE records[k].parent]
     IN /\ beam'=[k \in DOMAIN ids |-> generated[ids[k]]]
        /\ expressions'=[k \in DOMAIN ids |-> Eval(expressions[parents[k]],Bits(concepts[records[k].concept]),records[k].operation)]
        /\ ancestry'=[k \in DOMAIN ids |-> [parent |-> parents[k], concept |-> records[k].concept, operation |-> records[k].operation]] /\ previous'=beam
  /\ iteration'=iteration+1 /\ phase'="committed"
  /\ UNCHANGED <<neuron,rawConcepts,concepts,width,cap,eligible,generated,unique,representatives,selected>>
Continue == /\ phase="committed" /\ phase'="ready"
  /\ generated'= << >> /\ unique'= << >> /\ representatives'= << >> /\ selected'= << >>
  /\ eligible'= << >> /\ ancestry'= << >> /\ previous'= << >>
  /\ UNCHANGED <<neuron,rawConcepts,concepts,width,cap,iteration,beam,expressions>>
NoEligible == /\ phase="eligible" /\ Len(eligible)=0
  /\ beam'= << >> /\ expressions'= << >> /\ iteration'=iteration+1 /\ phase'="ready"
  /\ UNCHANGED <<neuron,rawConcepts,concepts,width,cap,eligible,generated,unique,representatives,selected,ancestry,previous>>
Stop == /\ phase="ready" /\ (Len(beam)=0 \/ iteration>=cap) /\ phase'="done"
  /\ UNCHANGED <<neuron,rawConcepts,concepts,width,cap,iteration,beam,expressions,eligible,generated,unique,representatives,selected,ancestry,previous>>
Done == /\ phase="done" /\ UNCHANGED vars
Next == Initialize \/ ChooseEligible \/ Generate \/ Deduplicate \/ Select \/ Commit \/ Continue \/ NoEligible \/ Stop \/ Done
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
FormulaVectorAgreement == \A k \in DOMAIN beam: expressions[k]=Bits(beam[k])
FlattenCoordinates == phase \in {"generated","deduplicated","selected"} =>
  \A p \in DOMAIN eligible,c \in DOMAIN concepts,o \in 1..3:
    LET r==((p-1)*Len(concepts)+c-1)*3+o IN
      Bits(generated[r])=Eval(expressions[eligible[p]],Bits(concepts[c]),o)
FirstRepresentative == phase \in {"deduplicated","selected","committed"} =>
  \A k \in DOMAIN unique : /\ generated[representatives[k]]=unique[k]
    /\ \A r \in DOMAIN generated : generated[r]=unique[k] => representatives[k]<=r
StableScoreOrder == phase="selected" =>
  /\ Len(selected)=Min({width,Len(unique)})
  /\ \A a,b \in DOMAIN selected : a<b =>
     (GT(unique[selected[a]],unique[selected[b]]) \/
       (EQ(unique[selected[a]],unique[selected[b]]) /\ representatives[selected[a]]<representatives[selected[b]]))
  /\ \A a \in DOMAIN selected,b \in (DOMAIN unique) \ {selected[k]:k \in DOMAIN selected}:
     GT(unique[selected[a]],unique[b]) \/
       (EQ(unique[selected[a]],unique[b]) /\ representatives[selected[a]]<representatives[b])
AncestryMatchesFirstRow == phase="committed" => \A k \in DOMAIN beam:
  \E p \in DOMAIN eligible,c \in DOMAIN concepts,o \in 1..3:
    /\ ((p-1)*Len(concepts)+c-1)*3+o=representatives[selected[k]]
    /\ ancestry[k]=[parent |-> eligible[p],concept |-> c,operation |-> o]
AncestryUsesSourceBeam == phase="committed" => \A k \in DOMAIN beam:
  Bits(beam[k])=Eval(Bits(previous[ancestry[k].parent]),Bits(concepts[ancestry[k].concept]),ancestry[k].operation)
IterationBound == iteration<=cap
EventuallyStops == <>(phase="done")
=============================================================================
