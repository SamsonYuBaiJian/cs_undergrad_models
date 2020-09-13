(define (domain logistic)
  (:predicates 	(package ?p)
                (truck ?t)
               	(location ?l)
    (road ?l ?l)
		(packageat ?p ?l)
    (truckat ?t ?l)
		(in ?p ?t))
 
(:action load
  :parameters
   (?pack
    ?truck
    ?loc)
  :precondition
  (and (package ?pack) (truck ?truck) (location ?loc)
  (truckat ?truck ?loc) (packageat ?pack ?loc))
  :effect (and (not (packageat ?pack ?loc)) (in ?pack ?truck)))

(:action unload
  :parameters
   (?pack
    ?truck
    ?loc)
  :precondition
  (and (package ?pack) (truck ?truck) (location ?loc)
  (truckat ?truck ?loc) (in ?pack ?truck))
  :effect (and (not (in ?pack ?truck)) (packageat ?pack ?loc)))

(:action move
  :parameters
   (?truck
    ?from
    ?to)
  :precondition
  (and (truck ?truck) (location ?from) (location ?to)
  (truckat ?truck ?from) (road ?from ?to))
  :effect (and (not (truckat ?truck ?from)) (truckat ?truck ?to))))