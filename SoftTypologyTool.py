#Soft Typology Tool v 0.2
# Charlie O'Hara
# 2/8/18
#This tool performs simulations of generational learning using a Generational Stability Model.
#The GUI allows input of tableau in the format used by OT-Help.
#Currently this tool performs MaxEnt learning with a fixed LearningRate across all constraints
#Initial Constraint Weights can be set
#Patterns to be examined are selected in app, currently all logically possible patterns are available.
#The number of iterations per generation, generations per run, and runs per pattern can be set.
#The simulations output:
# -Stability rates for each pattern
#   (% of runs where final generation produced the same form as the original pattern with at least .5 prob for all inputs
# -Plot of learning for the first generation in the first run of each pattern
# -Plot of generational change for the first run of each pattern
# -Plot of the resulting typology (What do patterns become?)
import pickle
import numpy as np
from Tkinter import *
from tkFileDialog import askopenfilename
from ttk import *
import collections
import time
import matplotlib
from datetime import date
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

import matplotlib.pyplot as plt



system="MaxEnt"
iffirstgen=True
ifgenmap=True
ifsoft=False
onehier=False
error=False
switch=False
noONSET=False
lastguy=False
#Import Tableau
inputorder=[]
"""tableaufile="test"
file=open(tableaufile,"r")

reader= file.readlines()
inputorder=[]
tableau={}
for l in range(1,len(reader)):
    splitline=reader[l].replace("\n","").split('\t')

    input=splitline[0]
    output=splitline[1]
    vect=map(int,splitline[3:])
    if input in tableau:
        tableau[input][output]=np.array(vect)
    else:
        tableau[input]={output:np.array(vect)}
        inputorder.append(input)
        print input"""
#print tableau
variables=[]
weightentries=[]

def importtableau():

    tableaufile=userfile.get()
    file = open(tableaufile, "r")

    global tableau
    reader = file.readlines()
    global inputorder
    if LOADING==False:
        inputorder = []
    outputorder=[]
    if LOADING==False:
        tableau = {}
    global weightentries
    constraints=reader[0].split()[3:]
    global start
    if LOADING==False:
        start=[]
    consname = tableaufile + "-cons"
    consfileinfo = {}

    def inweightpop():
        global weightentries
        weightentries = []
        initialweights = Tk()

        def saveinitial():
            global weightentries
            global start

            start = []
            for i in weightentries:
                start.append(float(i.get()))
            initialweights.destroy()

        for con in range(len(constraints)):
            Label(initialweights, text=constraints[con]).grid(row=con, column=0)
            if len(weightentries)<len(constraints):
                conweight = Entry(initialweights)
                weightentries.append(conweight)
            else:
                conweight=weightentries[con]
            conweight.grid(row=con, column=1)
            conweight.insert(END, start[con])
            #saveinitial()
        Button(initialweights, text="Save Initial Weights", command=saveinitial).grid()

    if LOADING==False:
        try:
            consfile = open(consname, "r")
            reader1 = consfile.readlines()

            for x in reader1:
                splitrow = x.split()
                consfileinfo[splitrow[0]] = splitrow[1:]
            for con in range(len(constraints)):
                start.append(float(consfileinfo["Weights"][con]))

        except IOError:
            errorpopup = Toplevel()
            Label(errorpopup, text="No file %s exists" % consname).grid()

            def errorcancel():
                errorpopup.destroy()
                return
            def setthem():
                errorpopup.destroy()
                consfileinfo["Weights"]=[1]*len(constraints)
                for con in range(len(constraints)):
                    start.append(float(consfileinfo["Weights"][con]))

                inweightpop()
            Button(errorpopup, text="Cancel", command=errorcancel).grid(row=1)
            Button(errorpopup, text="Set Weights", command=setthem).grid(row=2)


    #inweightpop()
    Button(master, text="Initial Weights", command=inweightpop).grid(row=0,column=5)
    if LOADING==False:
        for l in range(1, len(reader)):
            splitline = reader[l].replace("\n", "").split('\t')



            input = splitline[0]
            output = splitline[1]
            vect = map(int, splitline[3:])
            outputorder.append(output)
            if input in tableau:
                tableau[input][output] = np.array(vect)
            else:
                tableau[input] = {output: np.array(vect)}
                d=collections.OrderedDict()
                d[output]=np.array(vect)
                tableau[input] = d
                inputorder.append(input)
                #print input
    #tableaubox.insert(END,''.join(reader))
    n=0
    print inputorder
    global variables
    variables=[]
    for i in range(len(inputorder)):
        Button(master, text=inputorder[i]).grid(row=n+1, column=4)
        variables.append(IntVar())
        for j in range(len(tableau[inputorder[i]])):
            Radiobutton(master,text=tableau[inputorder[i]].keys()[j],variable=variables[i],value=j).grid(row=n+1,column=5)
            n=n+1
    Button(master, text="Use Pattern",command=savepattern).grid(row=0,column=4)
    print "Imported"
savedpatterns={}
savedlabels={}
delete=[]
this=0
langrow=0
def savepattern():
    pattern=""
    for x in range(len(variables)):
        #print inputorder
        pattern=pattern+tableau[inputorder[x]].keys()[variables[x].get()]
        pattern=pattern+"\t"
    global savedpatterns
    global savedlabels
    global this
    global langrow
    savedpatterns[this]=pattern
    button=Button(master, text="Delete", command=lambda this=this: delete(this))
    label=Label(master,text=pattern)
    savedlabels[this]=(button,label)
    button.grid(column=1)
    label.grid(row=button.grid_info()['row'],column=0)
    this=this+1
    langrow=langrow+1

def delete(x):
    global savedlabels
    global savedpatterns
    global langrow
    label,button=savedlabels[x]
    label.destroy()
    button.destroy()
    savedlabels.pop(x)
    savedpatterns.pop(x)
    langrow=langrow-1

def runningpopup(x):
    global popup
    global popupvars
    global cancelsims
    popup=Toplevel()
    lang_var=StringVar()
    lang_var.set(x)
    Label(popup, text="Running Right Now:").grid()
    Label(popup, textvariable=lang_var).grid()
    run_var=DoubleVar()
    run_var.set(1)
    runbar=Progressbar(popup,variable=run_var,maximum=runs)
    Label(popup,text="Runs: ").grid(row=3,column=0)
    runbar.grid(row=3,column=1)
    gen_var = DoubleVar()
    gen_var.set(0)
    genbar = Progressbar(popup, variable=gen_var, maximum=generations)
    Label(popup, text="Generations: ").grid(row=4, column=0)
    genbar.grid(row=4, column=1)
    it_var = DoubleVar()
    it_var.set(0)
    itbar = Progressbar(popup, variable=it_var, maximum=iterations)
    print iterations
    Label(popup, text="Iterations: ").grid(row=5, column=0)
    itbar.grid(row=5, column=1)
    popupvars=[lang_var,run_var,gen_var,it_var]
    popup.update_idletasks()
    Button(popup,text="Cancel", command=cancel).grid()
def updaterunner(l,r,g,i):
    clockcall=time.clock()
    global popupvars
    global popup
    popup.update()
    popupvars[0].set(l)
    popupvars[1].set(r)
    popupvars[2].set(g)
    popupvars[3].set(i)
    #print time.clock()-clockcall


def cancel():
    global cancelsims
    cancelsims=TRUE
    popup.destroy()
cancelsims=TRUE
langbuts=[]
LOADING=FALSE
def browsetableau():
    global tableauname
    tableauname=askopenfilename()
    tableaunameclean=tableauname.split("/").pop()
    resetEntries()
def load():
    global inputorder,LOADING,tableauname, start,savedpatterns, tableau,learningrate,generations,iterations,runs,bigtyp,gengraphs,itgraphs
    while len(savedpatterns)>0:
        delete(savedpatterns.keys()[0])
    filename=askopenfilename()
    loadee=file(filename,"rb")
    inputorder,tableauname,tableau,start, savedpatterns, learningrate,generations,iterations,runs,bigtyp,gengraphs,itgraphs= pickle.load(loadee)
    resetEntries()
    LOADING=TRUE
    importtableau()
    for index in savedpatterns.keys():
        pattern=savedpatterns[index]
        button = Button(master, text="Delete", command=lambda this=this: delete(this))
        label = Label(master, text=pattern)
        savedlabels[index] = (button, label)
        button.grid(column=1)
        label.grid(row=button.grid_info()['row'], column=0)
        index = index + 1
    runsims()
    LOADING=False
def runsims():
    langbuts=[]
    global cancelsims
    cancelsims=FALSE

    def savesim():
        filename = "tableau" + "lr" + "gens" + "iterations" + "runs" + "massiveFile"
        filename="Sim of "+tableauname
        f=file(filename,"wb")
        pickle.dump([inputorder,tableauname,tableau,start, savedpatterns, learningrate, generations, iterations, runs, bigtyp, gengraphs, itgraphs],f)

    def graphs():
        def buildlangs():

            global langbuts
            if graphv.get()=="typ":
                for x in langbuts:
                    x.destroy()
                graphtyp()
                langbuts=[]

            else:
                if len(langbuts) == 0:
                    dropdown = OptionMenu(graphpopup, v, itgraphs.keys()[0], *itgraphs.keys(),command=graphit)
                    #dropdown.grid()
                    langbuts.append(dropdown)
                    #for index in range(len(itgraphs)):
                     #   pat=itgraphs.keys()[index]
                      #  but=Radiobutton(graphpopup,text=pat,variable=v, value=pat,command=graphit)
                       # langbuts.append(but)
                    droplabel=Label(graphpopup,text="Pattern")
                    droplabel.grid(column=0,row=1)
                    langbuts.append(droplabel)
                    dropdown.grid(column=1,row=1)
                    dropruns=OptionMenu(graphpopup,runvar,1,*range(1,runs+1),command=graphit)
                    dropruns.grid(column=1,row=2)
                    droprunlabel=Label(graphpopup,text="Run")
                    droprunlabel.grid(column=0,row=2)
                    dropgens=OptionMenu(graphpopup,genvar,1,*range(1,generations+1),command=graphit)
                    dropgens.grid(column=3,row=2)
                    dropgenlabel=Label(graphpopup,text="Generation")
                    dropgenlabel.grid(column=2,row=2)
                    langbuts.append(dropgenlabel)
                    langbuts.append(dropgens)
                    langbuts.append(dropruns)
                    langbuts.append(droprunlabel)
                graphit()
        def graphtyp():
            f.clf()
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
            ind = np.arange(len(totaltyp))
            plotlist = []
            for i in range(len(bigtyp)):
                startlang = bigtyp.keys()[i]
                langplot = []

                endset = bigtyp[startlang]
                for endlang in totaltyp:
                    if endlang in endset:
                        langplot.append(endset[endlang])
                    else:
                        langplot.append(0)
                if i == 0:
                    oneplot = plt.bar(ind, langplot, .35, color=colors[i], label=startlang)
                    plotlist.append(oneplot)
                    totalplot = np.array(langplot)
                else:
                    oneplot = plt.bar(ind, langplot, .35, bottom=totalplot, color=colors[i], label=startlang)
                    plotlist.append(oneplot)

                    totalplot += np.array(langplot)

            plt.subplots_adjust(bottom=0.2,right=.62)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.xticks(ind, totaltyp.keys(), rotation=40)
            plt.xlabel("Pattern after Simulations")
            plt.ylabel("Count")
            #plt.tight_layout()
            canvas.show()
        def graphit(*hmm):
            f.clf()
            a=f.add_subplot(111)
            graphrun=runvar.get()-1
            graphgeneration=genvar.get()-1
            if graphv.get()=="it":
                arrdict=itgraphs
                xlabel="Iterations (x%s)"%iterational
            else:
                arrdict=gengraphs
                xlabel="Generations"
            lang=v.get()
            labels=arrdict[lang]["labels"]
            if graphv.get()=="it":
                plotted = a.plot(arrdict[lang]["array"][graphrun][graphgeneration])
            else:
                plotted=a.plot(arrdict[lang]["array"][graphrun])
            a.legend(plotted, labels, loc=0)
            plt.ylim([-.1, 1.1])
            plt.xlabel(xlabel)
            plt.ylabel("Probability")
            canvas.show()
        graphpopup = Toplevel()
        v=StringVar()
        v.set(itgraphs.keys()[0])
        runvar=IntVar()
        genvar=IntVar()
        genvar.set(0)
        runvar.set(0)
        graphv=StringVar()
        graphv.set("it")
        Radiobutton(graphpopup,text="Within Generation Learning",variable=graphv,value="it",command=buildlangs).grid(column=0,row=0)
        if gen.get()>=1:
            Radiobutton(graphpopup, text="Generational Change", variable=graphv, value="gen",command=buildlangs).grid(column=1, row=0)
        Radiobutton(graphpopup, text="Result Typology", variable=graphv, value="typ",command=buildlangs).grid(column=2, row=0)

        f=plt.figure(1)
        canvas=FigureCanvasTkAgg(f,master=graphpopup)
        canvas.show()
        canvas.get_tk_widget().grid(row=3,columnspan=3)
        toolbar_frame = Frame(graphpopup)
        toolbar_frame.grid(row=4, columnspan=3)
        toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
        toolbar.update()
        def smash():
            global langbuts
            graphpopup.destroy()
            langbuts=[]
        Button(graphpopup,text="Close",command=smash).grid()
        buildlangs()


    global bigtyp
    #Get the values from each of the buttons
    global LOADING
    global rentry
    global it
    global gen
    global lr
    global start
    global runs, iterations, generations, learningrate
    if LOADING==FALSE:
        bigtyp={}
    runs=int(rentry.get())
    iterations=int(it.get())
    generations=int(gen.get())
    learningrate=float(lr.get())
    runningpopup("")
    #t = threading.Thread(target=runningpopup(""))
    #t.start()
    global savedpatterns
    languages= [x.split() for x in savedpatterns.values()]
    #Test to see if a tableau has been imported, throw an error window if not.
    if len(tableau)==0:
        warning=Tk()
        Label(warning,text="No tableau was imported").grid()
    #Test to see if any patterns are being searched over, throw an error window if not
    elif len(savedpatterns)==0:
        warning=Tk()
        Label(warning,text="No patterns were selected").grid()
    else:#Run Simulations
        #Iterate over the languages saved
        if LOADING:
            global gengraphs
            global itgraphs
        else:
            gengraphs={}
            itgraphs={}
        if LOADING==FALSE:
            for l in languages:
                outputs = np.array(l)
                #Define the input output mappings
                language = {}
                for i in range(len(tableau.keys())):
                    language[inputorder[i]] = outputs[i]
                #Make a string of the language
                initlangname = ""
                for word in inputorder:
                    outword = language[word]
                    if word == outword:
                        initlangname = initlangname + word + " "
                    else:
                        initlangname = initlangname + outword + " "

                typology = {}
                genmapbyrun=[]
                firstgenbyrun=[]
                genbyrun=[]
                #Iterate over number of runs
                for r in range(runs):
                    w = start
                    genmap = []
                    firstgen = []
                    errorlist={}
                    genlist=[]
                    for x in inputorder:
                        errorlist[x]=0
                    #errorlist = {"tV": 0, "pV": 0, "kV": 0, "Vt": 0, "Vp": 0, "Vk": 0}
                    #Iterate over number of generations
                    for g in range(generations):
                        eachgen=[]
                        #Iterate over number of iterations
                        for i in range(iterations):
                            if cancelsims: break
                            #Update the popup
                            if i % iterational==0:
                                updaterunner(initlangname,r,g,i)
                            # Choose a random input
                            input = np.random.choice(tableau.keys(), 1)[0]
                            # Find observed output
                            if g > 0: #Select it from the teacher's grammar
                                observedoutput = output(teacherw, input)
                            else: # Select it from the initial language
                                observedoutput = language[input]
                            # Find the number of violations of the observed form

                            errorvios = tableau[input][observedoutput]
                            if error:
                                errorchance = 1
                                for j in range(len(errorrates)):
                                    # print 1+errorrates[i]*errorvios[i]
                                    errorchance *= (1 + errorrates[j] * errorvios[j])
                                errorchance = 1 - errorchance
                                # print g, language
                                # print errorrates
                                # print observedoutput,errorvios
                                # print input,errorchance
                                rand = np.random.random()
                                # print input
                                if rand < errorchance:
                                    # print "%s <%s"%(rand,errorchance)
                                    if switch:
                                        observedoutput1 = \
                                        np.random.choice(([k for k in tableau[input].keys() if k != observedoutput]), 1)[0]
                                        # print "An error occurred %s to  %s" %(observedoutput,observedoutput1)
                                        observedoutput = observedoutput1
                                    else:

                                        continue
                                        #  else:
                                        # print i,observedoutput
                            observedvios = tableau[input][observedoutput]
                            # Find learner output
                            expectedoutput = output(w, input)
                            expectedvios = tableau[input][expectedoutput]

                            # Find training ERC
                            ERC = observedvios - expectedvios
                            # Update Weights
                            w1 = w + learningrate * ERC
                            # prevent weights from going negative
                            if (np.array(np.where(w1 < 0)).size > 0):
                                w1[np.where(w1 < 0)] = .1
                                w1[np.where(w1 < 0)] = w[np.where(w1 < 0)]
                            if lastguy:
                                oldprob = probtableau(w)
                                newprob = probtableau(w1)
                                if g == 1:
                                    lastone = lastlist[initlangname].keys()[0]
                                    if oldprob[lastone][lastone] < .9:
                                        if newprob[lastone][lastone] > .9:
                                            # print "hey!"
                                            lastlist[initlangname][lastone].append(i)
                                            # print "%s   %s"%(oldprob[lastone][lastone],newprob[lastone][lastone])

                            w = w1

                            # Save the iteration to the firstgeneration
                            if iffirstgen:
                                if i % iterational ==0:
                                    probs = probtableau(w)
                                    corrects = []
                                    # print w
                                    for x in inputorder:
                                        corrects.append(probs[x][language[x]])

                                    for c in corrects:
                                        eachgen.append(c)
                                if g == 0:
                                    if (observedvios != expectedvios).any():
                                        # print input
                                        errorlist[input] = errorlist[input] + 1
                                    if i % iterational == 0:
                                        probs = probtableau(w)
                                        corrects = []
                                        # print w
                                        for x in inputorder:
                                            corrects.append(probs[x][language[x]])

                                        for c in corrects:
                                            firstgen.append(c)

                        teacherw = w
                        w = start
                        genlist.append(eachgen)
                        #if ifgenmap & r == 0:
                        if ifgenmap:
                            probs = probtableau(teacherw)
                            gencorrect = []
                            for x in inputorder:
                                gencorrect.append(probs[x][language[x]])
                            for m in gencorrect:
                                genmap.append(m)
                    probs = probtableau(teacherw)
                    genmapbyrun.append(genmap)
                    firstgenbyrun.append(firstgen)
                    genbyrun.append(genlist)

                    # Save the probability of each correct form at this generation

                    # print probs
                    outputlang = {}
                    langname = ""
                    for word in inputorder:
                        outword = max(probs[word], key=probs[word].get)
                        outputlang[word] = [outword]
                        if word == outword:
                            langname = langname + word + " "
                        else:
                            langname = langname + outword + " "
                            # print langname
                    try:
                        typology[langname] += 1
                    except KeyError:
                        typology[langname] = 1
                #Make labels for the legend
                labels = []
                for x in inputorder:
                    labels.append("%s -> %s" % (x, language[x]))

                #for i in range(len(firstgenbyrun)):

                #Save results of simulations as arrays for making graphs.
                #firstgarray = np.reshape(firstgen, newshape=(len(firstgen) / len(inputorder), len(inputorder)))
               # print firstgarray
                print len(firstgenbyrun[0])
                print len(genbyrun[0][0])
                firstgarray=map(lambda x: np.reshape(x,newshape=(len(x)/len(inputorder),len(inputorder)) ),firstgenbyrun)
                genlistarray=map(lambda y: map(lambda x: np.reshape(x,newshape=(len(x)/len(inputorder),len(inputorder)) ),y),genbyrun)
                print(len(genlistarray),len(genlistarray[0]),len(genlistarray[0][0]))

                #np.reshape(firstgenbyrun
                itgraphs[initlangname]={"array":genlistarray,"labels":labels}
               # gmarray = np.reshape(genmap, newshape=(len(genmap) / len(inputorder), len(inputorder)))
                #print "hello?"

                gmarray = map(lambda x: np.reshape(x, newshape=(len(x) / len(inputorder), len(inputorder))),
                                  genmapbyrun)

                gengraphs[initlangname]={"array":gmarray,"labels":labels}
                bigtyp[initlangname] = typology
                print typology
                print bigtyp
        print bigtyp

        # Create total typology from all start languages
        stability = {}
        totaltyp = {}
        for x in bigtyp:
            set = bigtyp[x]
            try:
                stability[x] = float(bigtyp[x][x]) / runs
            except KeyError:
                stability[x] = 0
            for y in set:
                try:
                    totaltyp[y] += set[y]
                except KeyError:
                    totaltyp[y] = set[y]
        print totaltyp
        if lastguy:
            print lastlist
            average = {}
            standdev = {}
            for x in lastlist:
                for y in lastlist[x]:
                    l = lastlist[x][y]
                    average[x] = sum(l) / float(len(l))
                    standdev[x] = np.std(l)
            print average
            print standdev
        print "errorrates: %s" % errorrates
        if (cancelsims==False):
            stabilitypopup=Toplevel()
            Label(stabilitypopup,text="Stability Rates").grid(row=0,columnspan=2)
            for x in range(len(stability.keys())):
                pattern=stability.keys()[x]
                stabrate=stability[pattern]
                Label(stabilitypopup,text=pattern).grid(row=x+1,column=0)
                Label(stabilitypopup,text=stabrate).grid(row=x+1,column=1)
            Button(stabilitypopup, text="Graphs",command=graphs).grid()
            Button(stabilitypopup,text="Save",command=savesim).grid()
            Button(stabilitypopup, text="Quit", command=quit).grid()
            print "stability: \n%s" % stability


#tableau={"A":{"a":np.array([-1,0,-1,0,-1]),"b":np.array([0,-1,-1,0,-1])}}
if ifsoft:
    tableau={"tV":{"tV":np.array([0,0,-1,-1,0,0,-1,0,0]),"V":np.array([0,0,0,0,-1,-1,0,0,0])},
             "pV": {"pV": np.array([0,-1,0,-1,0,0,-1,0,0]), "V": np.array([0,0,0,0,-1,-1,0,0,0])},
             "kV": {"kV": np.array([-1,0,0,-1,0,0,-1,0,0]), "V": np.array([0,0,0,0,-1,-1,0,0,0])},
             "Vt": {"Vt": np.array([0,0,-1,-1,0,0,0,-1,0]), "V": np.array([0,0,0,0,-1,0,0,0,-1])},
             "Vp": {"Vp": np.array([0,-1,0,-1,0,0,0,-1,0]), "V": np.array([0,0,0,0,-1,0,0,0,-1])},
             "Vk": {"Vk": np.array([-1,0,0,-1,0,0,0,-1,0]), "V": np.array([0,0,0,0,-1,0,0,0,-1])}
             }
else:
    if onehier:
        tableau = {"tV": {"tV": np.array([0, 0, 0,0,0,-1, 0]), "V": np.array([0, 0, 0,0,0,0, -1])},
                   "pV": {"pV": np.array([0, 0,0,0,-1, -1, 0]), "V": np.array([0, 0, 0,0,0,0, -1])},
                   "kV": {"kV": np.array([0,0,0,-1, -1, -1, 0]), "V": np.array([0, 0, 0,0,0,0, -1])},
                   "Vt": {"Vt": np.array([0,0,-1,-1, -1, -1, 0]), "V": np.array([0, 0, 0,0,0,0, -1])},
                   "Vp": {"Vp": np.array([0,-1,-1,-1, -1, -1, 0]), "V": np.array([0, 0, 0,0,0,0, -1])},
                   "Vk": {"Vk": np.array([-1,-1,-1,-1, -1, -1, 0]), "V": np.array([0, 0, 0,0,0,0, -1])}}
    else:
        tableau={"tV":{"tV":np.array([0,0,-1,0,0,0]),"V":np.array([0,0,0,-1,-1,0])},
             "pV": {"pV": np.array([0, -1, -1, 0, 0, 0]), "V": np.array([0, 0, 0, -1, -1, 0])},
             "kV": {"kV": np.array([-1, -1, -1, 0, 0, 0]), "V": np.array([0, 0, 0, -1, -1, 0])},
             "Vt": {"Vt": np.array([0, 0, -1, 0, 0, -1]), "V": np.array([0, 0, 0, -1, 0, 0])},
             "Vp": {"Vp": np.array([0, -1, -1, 0, 0, -1]), "V": np.array([0, 0, 0, -1, 0, 0])},
             "Vk": {"Vk": np.array([-1, -1, -1, 0, 0, -1]), "V": np.array([0, 0, 0, -1, 0, 0])}
             }
        if noONSET:
            tableau = {"tV": {"tV": np.array([0, 0, -1, 0, 0, 0]), "V": np.array([0, 0, 0, -1, 0, 0])},
                       "pV": {"pV": np.array([0, -1, -1, 0, 0, 0]), "V": np.array([0, 0, 0, -1, 0, 0])},
                       "kV": {"kV": np.array([-1, -1, -1, 0, 0, 0]), "V": np.array([0, 0, 0, -1, 0, 0])},
                       "Vt": {"Vt": np.array([0, 0, -1, 0, 0, -1]), "V": np.array([0, 0, 0, -1, 0, 0])},
                       "Vp": {"Vp": np.array([0, -1, -1, 0, 0, -1]), "V": np.array([0, 0, 0, -1, 0, 0])},
                       "Vk": {"Vk": np.array([-1, -1, -1, 0, 0, -1]), "V": np.array([0, 0, 0, -1, 0, 0])}
                       }
tableau={}

errorrates=np.array([0.03,.1,0,0,0,.2])
errorrates=np.array([0,0,0,0,0,0])
#Return the probability for each candidate given the weightings
def probtableau(weight):
    probdist={}
    for input in tableau:
        inptab=tableau[input]
        harmdict={}
        # find harmony score for each candidate
        for candidate in inptab:
            harmony = inptab[candidate].dot(weight)
            harmdict[candidate] = harmony

        # Find Winner (HG)
        if (system == "HG"):
            print max(harmdict)
            return max(harmdict)
        # Find winner MaxEnt
        elif (system == "MaxEnt"):
            probdict = {}
            z = sum(np.exp(harmdict.values()))
            for candidate in inptab:
                probdict[candidate] = (np.exp(harmdict[candidate]) / z)
        probdist[input]=probdict
    return probdist

#Generate an output given the current weighting (weight) and input form.
def output(weight,input):
    inptab=tableau[input]
    harmdict={}
    #find harmony score for each candidate
    for candidate in inptab:

        harmony=inptab[candidate].dot(weight)
        harmdict[candidate]=harmony

    #Find Winner (HG)
    if(system=="HG"):
        print max(harmdict)
        return max(harmdict)
    #Find winner MaxEnt
    elif(system=="MaxEnt"):
        probdict={}
        z= sum(np.exp(harmdict.values()))
        for candidate in inptab:
            probdict[candidate]=(np.exp(harmdict[candidate])/z)
        return np.random.choice(probdict.keys(),1,p=probdict.values())[0]

def resetEntries():
    def replaceText(entry,text):
        entry.delete(0,END)
        entry.insert(END,text)
    tableaunameclean = tableauname.split("/").pop()
    replaceText(userfile,tableaunameclean)
    replaceText(rentry,runs)
    replaceText(it,iterations)
    replaceText(gen,generations)
    replaceText(lr,learningrate)

#Default rate of learning at each update
learningrate=.05
#Default number of generations per run
generations = 25
#Default number of iterations per generation
iterations = 3600

if onehier: iterations=5500
if ifsoft:
    learningrate=.05
    iterations=2000
iterational=100
#number of runs per language per simulation
runs=1

# Generate the set of inputs
#inputorder="tV pV kV Vt Vp Vk".split()
#inputorder=tableau.keys()
#print inputorder

#language={"tV":"tV","pV":"pV","kV":"V","Vt":"Vt","Vp":"Vp","Vk":"V"}
#Generate the logically possible set of languages
tableauname="test"


languages=["tV pV kV Vt V V".split(),"tV pV kV Vt Vp Vk".split()]


master = Tk()

userfile = Entry(master, text="File: ")
#userfile.insert(END, 'test')
userfile.grid(row=0,column=1)
rentry=Entry(master)
rentry.grid(row=1,column=1)
Label(master,text="File: ").grid(row=0,column=0)
Button(master,text="Browse",command=browsetableau).grid(row=0,column=2)
Label(master, text="Runs: ").grid(row=1,column=0)
#rentry.insert(END,runs)
it=Entry(master)
it.grid(row=2,column=1)
#it.insert(END, iterations)
Label(master, text="Iterations: ").grid(row=2,column=0)
gen=Entry(master)
gen.grid(row=3,column=1)
#gen.insert(END, generations)
Label(master, text="Generations: ").grid(row=3,column=0)
lr=Entry(master)
lr.grid(row=4,column=1)
Label(master, text="Learning Rate: ").grid(row=4,column=0)
#lr.insert(END,learningrate)
resetEntries()
b= Button(master, text="Import File", command=importtableau)
b.grid(row=0,column=3)
FIRSTGEN= IntVar()
#firster= Checkbutton(master, text="Plot First Generation Learning?", variable=FIRSTGEN)
#firster.grid(row=1, column=2)
runbut=Button(master,text="Run", command=runsims)
runbut.grid()
quitbut=Button(master,text="Quit",command=quit)
quitbut.grid()
loadbut=Button(master,text="Load Simulation",command=load)
loadbut.grid()
#tableaubox = Text(master, width=100)
#tableaubox.grid(rowspan=1, columnspan=3)
mainloop()
def quit():
    master.destroy()