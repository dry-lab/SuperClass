#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;
use Encode qw(encode decode);
#############################################################################################
#
# Script name : simulator2exp.pl
# -----------
# Dev environment : - Ubuntu 14.04 x64
# ---------------   - perl, v5.18.2 built for x86_64-linux-gnu-thread-multi
#
#############################################################################################

my $enc = 'utf-8';
my $rootFolder = $ARGV[0];
my %filesHash ;

print STDERR "Entered ".$rootFolder."\n";

opendir my $dir, $rootFolder or die "Cannot open directory: $!";
	my @folders = readdir $dir;
closedir $dir;

@folders = grep(!/^\./, @folders);

foreach my $fold (@folders){

	opendir my $dir, "$rootFolder/$fold/" or die "Cannot open directory: $!";
		my @files = readdir $dir;
	closedir $dir;

	# my $out = "$fold"."_alltracks.txt";
	print STDERR "Entered ".$fold."\n";
	@files = grep(!/^\./, @files);
	
	# system "cat '$rootFolder/$fold/'* > '$out'";
	foreach my $file (@files){
		# print "$rootFolder/$fold/$file\n";
		open my $handle, '<', "$rootFolder$fold/$file";
			my @lines = <$handle>;
		close $handle;
		foreach my $line (@lines){
			$line = encode( $enc, $line ); 
			my @columns = split(/\s/, $line);
			# print Dumper @columns;
			$filesHash{"$fold"}{$file}{$columns[0]}{"NbPoints"}=$columns[1];
			$filesHash{"$fold"}{$file}{$columns[0]}{"Diffusion_Coefficient"}=$columns[2];
			$filesHash{"$fold"}{$file}{$columns[0]}{"MSD_0"}=$columns[3];
		}
	}
}
# print Dumper %filesHash;

hash2csv(\%filesHash);

sub hash2csv{
	my ($hash)=@_;
	my %hash = %$hash;
	my @per_image;
	my @per_object;
	my @per_point;

	my $well=0;
	my $pn = 0;
	my $imgNumber=0;
	my $objNumber=0;
	foreach my $key (keys(%hash)){
		$well++;
		print $key."\t".$well."\n"; 
		foreach my $file (keys($hash{$key})){
			$imgNumber++;
			my $waveTraceId=0;
			my $image_line = $imgNumber.","."1,".$well.",".$key.","."NA,"."NA,"."NA,"."NA,"."NA,"."NA\n";
			push(@per_image, $image_line);
			foreach my $track (keys($hash{$key}{$file})){
				$waveTraceId++;
				$objNumber++;
				my $length = $hash{$key}{$file}{$track}{"NbPoints"};
				my $object_line = $imgNumber.",".$waveTraceId.",".$objNumber.","."NA,"."NA,".$hash{$key}{$file}{$track}{"Diffusion_Coefficient"}.",".$hash{$key}{$file}{$track}{"MSD_0"}.","."NA,".$length."\n";
				push(@per_object, $object_line);
				
				for(my $i = 1; $i<=$length; $i++) {
					my $pn++;
					my $point_line = $pn.",".$waveTraceId.",".$imgNumber.",".$i.","."NA,"."NA\n";
					push(@per_point, $point_line);
				}
			}
		}
	}

my $filename = 'per_image.csv';
open(my $fh, '>', $filename) or die "Could not open file '$filename' $!";
foreach my $line (@per_image){
	print $fh $line;
}
close $fh;
print "done $filename\n";


my $filename2 = 'per_object.csv';
open(my $fh2, '>', $filename2) or die "Could not open file '$filename' $!";
foreach my $line (@per_object){
	print $fh2 $line;
}
close $fh2;
print "done $filename2\n";

my $filename3 = 'per_point.csv';
open(my $fh3, '>', $filename3) or die "Could not open file '$filename' $!";
foreach my $line (@per_point){
	print $fh3 $line;
}
close $fh3;
print "done $filename3\n";




}

