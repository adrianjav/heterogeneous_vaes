#!/bin/bash

echo -n "{'probabilistic model': ['"

first=true
for line in $( tr '\r' '\n' < "$1" | tail -n +2 | cut -d, -f 1,2); do
	type=$(echo $line | cut -d, -f 1)
	dim=$(echo $line | cut -d, -f 2)

	if ! $first ; then
		echo -n "', '"
	fi
	first=false

	if [ "$type" = "pos" ] ; then
		echo -n "lognormal"
	elif [ "$type" = "real" ] ; then
		echo -n "normal"
	elif [[ "$type" = "cat" ]] || [[ "$type" = "ordinal" ]]; then
	    if [ "$dim" = "2" ] ; then
		    echo -n "bernoulli"
            if [ "$2" = "--gamma-trick" ] ; then
                echo -n "*"
            fi
		else
		    echo -n "categorical($dim)"

		    if [[ "$2" = "--gamma-trick" ]] ; then
                echo -n "*"
            elif [[ "$2" = "--bern-trick" ]] ; then
            	echo -n "+"
            fi
		fi
	elif [ "$type" = "count" ] ; then
		echo -n "poisson"
		if [ "$2" = "--gamma-trick" ] ; then
		    echo -n "*"
		fi
	fi
done
echo -n "'], "

first=true
count=0
echo -n "'categoricals': ["
for line in $( tr '\r' '\n' < "$1" | tail -n +2 | cut -d, -f 1,2); do
	type=$(echo $line | cut -d, -f 1)
	dim=$(echo $line | cut -d, -f 2)

	if [[ "$type" = "cat" ]] || [[ "$type" = "ordinal" ]]; then
		if ! $first ; then
            echo -n ", "
        fi
        first=false

		echo -n "$count"
	elif [ "$type" = "ordinal" ] ; then
		if ! $first ; then
            echo -n ", "
        fi
        first=false

		echo -n "$count"
	fi
	count=$((count+1))
done
echo "]}"
