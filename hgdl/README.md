## HGDL implementation:
What each part does:
* hgdl.py - holds the client, info, and runs the whole show
* info.py - holds the args, deflated functions, and results
* results.py - holds the minima, presents shiny results
* bump.py - defines the bump function
* global\_methods: - all global methods go here
	* run\_global.py - runs each global step
	* genetic.py - defines a single genetic algorithm step
	* generic\_global.py - user can make a single file and modify run\_global easily to use custom code
* local\_methods: - hold all local processes
	* run\_local.py - same as global but uses dask distributed to parallelize. checks for redundant minima
	* newton.py - same as global scheme. each local method should just describe a single local optimizer

Code philosophy:
I try to make this code easy to read for the 3 people that will use it:
* the end user that doesn't care about anything but getting some optima quickly - make the API easy
* the user that wants to use a specicial doohicky for their local/global method - make the API flexible to taking these
* the user that wants to extend HGDL - make the package structure simple

Currently, I think both marcus and I have the code so that users 1 and 2 are happy. I would want user number 3 to also be happy. To me, this means that there should be few imports, few lines, and simple logic. This is what I do by having this tree structure:
 |
HGDL
 | - client
 | - info -> results
 | - futures - each future takes an info object and returns an info object. this means that the first future takes the initial info and then subsequent futures take the previous future. so the futures are [run\_epoch(info\_0), run\_epoch(futures[0]), run\_epoch(futures[1]), ..]

run epoch:
 - run\_global - run global process
 - run\_local - run local methods

Having the file system represent the layout of the method itself seems logical to me. putting everything together into a bunch of files with bizarre data structures seems like insanity to me. instead of having everything created in some sort of modified list that is passed to dask Variables and so on, have classes (info, results) that provide the interface you want.


