#!/usr/bin/perl

#############################################################################################
#
# Script name : simulationFileGenerator.pl
# -----------
# Dev environment : - Ubuntu 14.04 x64
# ---------------   - perl, v5.18.2 built for x86_64-linux-gnu-thread-multi
#
#############################################################################################

## Dependancies
use strict;
use warnings;
use Data::Dumper;

# my ()=@ARGS[];
# 

my $D_outsideContact = 0.0;
my $D_insideContact = 0.0;
my $k_on = 0.00;
my $k_off = 0.00;
my $D_trapped = 0.001;
my $k_on_fluo = 0.05;
my $k_off_fluo = 2.45;
my $N_planes = 1000;
my $N_repetition = 10;
my $N_particles = 1000;


for (my $j=0; $j<20; $j++){
	$k_on += 0.25;
	$k_off += 0.25;
	$D_outsideContact = 0.0;
	$D_insideContact = 0.0;

for (my $i=0; $i<=9; $i++){

	$D_outsideContact += 0.05;
	$D_insideContact += 0.05;
	# $k_on += 0.2;
	# $k_off += 0.2;
	# $D_trapped += 0.001;
	# $k_on_fluo = 2.45;
	# $k_off_fluo = 0.05;
	# $N_planes = 1000;
	# $N_repetition = 10;
	# $N_particles = 1000;

	my $string = qq(/*************************************************************/
/***********        SIMULATION PARAMETERS      ***************/
/*************************************************************/

**simulation mode** (to be selected in : {SPT, FRAP, FCS})
simulation_mode: SPT

**Number of particles**
N_particles: $N_particles

**Free Diffusion**
D_outsideContact: $D_outsideContact
D_insideContact: $D_insideContact

**Trapping**
k_on: $k_on
k_off: $k_off
D_trapped: $D_trapped

**Fluorescent parameters**
k_on_fluo: $k_on_fluo
k_off_fluo: $k_off_fluo
fcs_beam_sigma[um]: 0.063
fcs_beam_max_intensity: 1.0
fcs_noise_cutOff[%]: 50

**Temporal aspects**
N_presequ: 100
N_planes: $N_planes
N_frap: 0
dt_sim: 0.020

**Spatial aspects**
pixel_size[um]: 0.063

**Simulation aspects**
N_repetition: $N_repetition


/*************************************************************/
/*************************************************************/
/*************************************************************/);

	mkdir "SPT_FILES";

	my $filename = 'SPT_FILES/SPT_'.$j."_".$i.'.txt';
	open(my $fh, '>', $filename) or die "Could not open file '$filename' $!";
	print $fh $string;
	close $fh;
	print "create $filename\n";
}
}
print "Done\n";