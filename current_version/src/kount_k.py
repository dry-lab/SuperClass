#!/usr/bin/env python3.5

import argparse, gzip, sys
import numpy as np

#global definitions
transition={"A":0, "a":0, "C":1,"c":1, "G":2,"g":2,"T":3,"t":3}

def parse_function():
    parser=argparse.ArgumentParser(description="Calculates k-mer frequency from a given fasta file(s).")
    parser.add_argument('-f', '--file', dest="infile", type =str, help="fasta file for the sequences (left empty to read from stdin)")
    parser.add_argument("-k", "--k-mer", dest="kmer", type=int, help="kmer")
    parser.add_argument("-l", "--label", dest="label", type=int, help="label")
    parser.add_argument("-o", "--out-file",dest="outfile",type=str,help="name of the resulting file")

    args=parser.parse_args()
   
    if len(sys.argv)==1:
        # display help message when no args are passed.
        parser.print_help()
        sys.exit(1)

    return(args)            #return the args

def init_kmer_array(kmer):
    #print("Creating k-mer array")
    nk=4**kmer
    kmer_array=np.zeros(nk)
    return(kmer_array)

def kount_kmer(seq,kmer_array,k,label):
    #print("Profiling", id, "with", len(seq),"nucleotides.")
    n=0
    kmer_array.fill(0)
    #print(kmer_array)
    while n < len(seq)-k+1:
        f=int(0)
        r=int(0)
        for i in range(k):
            flag=1
            nuc=seq[n+i]
            try:
                f=f<<2|transition[nuc]
                r|= ((3-transition[nuc])<<(2*i))
            except:
                flag=0
                n=n+i+1
                break
        if flag==1:
            #print(n,f,kmer_f, r,kmer_r)
            kmer_array[f]+=1
            kmer_array[r]+=1
            n=n+1
    return(np.append(label,kmer_array)) 

def main():
    
    args=parse_function()
    infile=args.infile
    kmer=args.kmer
    label=args.label
    outfile=args.outfile

    kmer_array=init_kmer_array(kmer)
   
    try:
        extension=infile[-2:]
        if extension=="gz":
            print(infile)
            seq_file=gzip.open(infile,"r")
        else:
            seq_file=open(infile, 'r')
    except:
        seq_file=open("/dev/stdin","r")
    #print(seq_file)

    line=seq_file.readline()
    #print(line)
    if line[0]!=">":
        print("Not a fasta file")
        exit()
    output=open(outfile,'w')
    #print("%K-MER:",kmer,file=output,sep="", end="\n")

    while line!='':

        if line[0]==">":
            seq=''
            id=line.split()[0][1:]
            line=seq_file.readline().rstrip("\n")
        if line[0]!=">":
            seq=seq+line
            line=seq_file.readline().rstrip("\n")
        if line=="":
            #kmer_array=init_kmer_array(kmer)
            seq_ends=seq[0:10]+seq[-10:]
            array_k=kount_kmer(seq_ends,kmer_array,kmer,label)
            array_k.tofile(output,sep=":",format="%s")
            print(file=output)
            break
        if line[0]==">":
            #kmer_array=init_kmer_array(kmer)
            seq_ends=seq[0:10]+seq[-10:]
            array_k=kount_kmer(seq_ends,kmer_array,kmer,label)
            array_k.tofile(output,sep=":",format="%s")
            print(file=output)

    output.close()

if __name__ == '__main__':                  #the main function
    main()

