README on how to parse dbd files
Author: Gregory Burgess
Date:   4/25/21

The purpose of this README is to instruct the user how to use the binary executable dbd2asc (provided by Teledyne) and the parse_dbd executable (written by Gregory Burgess) to parse raw dbd files from the slocum glider into human readable asci files.
*Note: this was performed on UBUNTU Linux machine

To start, assemble a file structure as follows:
Directory
	--> dbd2asc
	--> parse_dbd
	--> unit_770-2021-099-16-1.dbd
	--> unit_770-2021-099-16-2.dbd
	--> etc.
	
*Make sure you set the correct priveleges for the executable files (chmod +x dbd2as)

Run ./parse_dbd

May have to run twice to get no errors due to necessity for dbd2asc to generate a cache repository in the directory. 

The output will be N number of .asc files for N number of dbd files with the same filename. 

May need to place .cac file from glider in generated /cache folder.
