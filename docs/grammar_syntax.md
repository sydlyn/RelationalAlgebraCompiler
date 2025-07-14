# Relational Algebra Compiler Syntax

## Overview
This is a () focused language. All queries must be wrapped in ().

Notes: 
- {} indicates a subscript

Supported Functions: 
- selection
- projection
- ...

Each function is described in more detail below:

### Selection

- /sigma
- /selection
- /select
- /s

example: 

(/sigma {age> 30} Students) => select * from Students where age > 30
(/selection_(age> 30) Students) => select * from Students where age > 30


### Projection
(/pi_{name, age} Students) => select distinct name, age from students
(/projection_{name, age} Students)=> select distinct name, age from students
(/pi^d_{name,age} Students) => select name, age from Students
(/projection^d_{name,age} Students)  => select name, age from Students
(/pi_{name, age+2 -> newAge} Students) => select distinct name, age+2 as newAge from Students
(/sigma_(age> 30) Students) => select * from Students where age > 30
(/selection_(age> 30) Students) => select * from Students where age > 30
(T1 /union T2) => select * from T1 union T2
(A /union^d B) => select * from A union all B
(A /intersect B)  => set intersection
(A /interesect^d B) => bag intersection
(A /difference  B) => set difference
(A /difference^d B) => bag difference
(A /X B) => select * from A, B
(A /cross_product B) => select * from A, B
(C1 /inner_join  C2) => select * from A natural join B
(C1 /inner_join_{C1.A=C2.B} C2) => select * from C1  join C2 on C1.A=C2.B //theta join, keeps duplicate column
(C1 /inner_join_{A,B} C2) => select * from C1  join C2 on C1.A=C2.A and C1.B=C2.B //theta join, removes duplicate columns
can use /join instead of /inner_join
(C1 /left_outer_join C2) => left outer join of two tables on common attributes that are removed from result
(C1 /left_outer_join_{C1.A=C2.B} C2) => left outer join on join condition, no attributes removed
(C1 /left_outer_join_{A,B} C2) =>  left outer join on A,B, attributes removed from result
(C1 /right_outer_join C2) => right outer join of two tables on common attributes that are removed from result
(C1 /right_outer_join_{C1.A=C2.B} C2) => right outer join on join condition, no attributes removed
(C1 /right_outerjoin_{A,B} C2) =>  right outer join on A,B, attributes removed from result
(C1 /full_outer_join  C2) => full outer join of two tables on common attributes that are removed from result
(C1 /full_outer_join_{C1.A=C2.B}  C2) => full outer join on join condition, no attributes removed
(C1 /full_outerjoin_{A,B} C2) =>  full outer join on A,B, attributes removed from result
(C1 /left_semi_join C2) => left semi join of two tables on common attributes
(C1 /left_semi_join_{C1.A=C2.B} C2) => left semi join on join condition
(C1 /left_semi_join_{A,B}  C2) =>  left semi join on A,B
(C1 /right_semi_join C2) => left semi join of two tables on common attributes
(C1 /right_semi_join_{C1.A=C2.B} C2) => left semi join on join condition
(C1 /right_semi_join_{A,B} C2) =>  left semi join on A,B
(C1 /divide C2) => division
/rho(T1, T2 /join T3) => creates table T1
/rename(T1, T2 /join T3) => creates table T1
(/delta T1) => select distinct * from T1
(/remove_duplicates T1) => select distinct * from T1
(/gamma_{A,B}{sum(C)} R} => select A,B, sum(C) from R group by A, B
(/group_{A,B}{sum(C)} R} => select A,B, sum(C) from R group by A, B
(/tau_{A asc, B asc} R} => select * from R order by A asc, B asc
(/sort_{A asc, B asc} R} => select * from R order by A asc, B asc

