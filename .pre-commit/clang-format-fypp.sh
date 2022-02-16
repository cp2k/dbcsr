#!/usr/bin/env bash

# clang-format change FYPP directives from "#: <directive>". Need to revert the change.

function main()
{
    local files=""
    for i in "$@"; do
	case $i in
	    -*|--*)
	    ;;
	    *)
		files+="$i "
	    ;;
	esac
    done

    clang-format "$@"
    # Fix FYPP directives
    for i in "${files}"; do
	sed -i "" 's/# : /#:/g' $i
    done
}

main "$@"
