%%
%% This is file `sangerposter.cls'
%% It was forked from Reinhold Kainhofer's baposter package on 7th Feb 2023 by Jack Fraser-Govil
%%
%% Licence: GPL
\ProvidesClass{sangerposter}[2023/02/07 v0.1 sangerposter class]
\NeedsTeXFormat{LaTeX2e}[1995/06/01]
\LoadClass{article}
\typeout{baposter: Brian Amberg, 2007, 2008, 2009, 2010, 2011 | http://www.brian-amberg.de/uni/poster/}
\typeout{baposter: Reinhold Kainhofer, 2011 | http://reinhold.kainhofer.com/}
\typeout{sangerposter: Jack Fraser-Govil, 2023}
%% Define lengths only once on inclusion, such that we can make multiple posters
\newlength{\sangerposter@basepaperwidth}
\newlength{\sangerposter@basepaperheight}
\newlength{\sangerposter@basemargin}
\newlength{\sangerposter@finalpaperwidth}
\newlength{\sangerposter@finalpaperheight}
\newlength{\sangerposter@finalmargin}
\newlength{\headerheight}%
\newlength{\colwidth}%
\newlength{\colheight}%
\newlength{\sangerposter@@colspacing}%
\newlength{\sangerposter@box@@cornerradius}%
\newlength{\sangerposter@box@@boxheaderheight}%
\newlength{\sangerposter@box@@boxpadding}%
\newlength{\boxstartx}%
\newlength{\boxstarty}%
\newlength{\boxwidth}%
\newlength{\boxheight}%
\newlength{\sangerposter@titleimage@left@width}%
\newlength{\sangerposter@titleimage@right@width}%
\newlength{\sangerposter@titleimage@textwidth}%
\newbox\sangerposter@box@content%
\newbox\sangerposter@titleimage@left%
\newbox\sangerposter@titleimage@title%
\newbox\sangerposter@titleimage@right%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Packages
%-------------------------------------------------------------------------------
% The only ``weird'' dependency of this package is pgf. All the rest should be
% installed on any decent system.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\typeout{Use Packages}
\RequirePackage{xkeyval}
\RequirePackage{fontspec} %% Sanger inclusion - forces use of XeLaTeX but enables custom fonts
\RequirePackage{calc}
\usepackage{lmodern}
%\RequirePackage[cmyk]{xcolor}
\RequirePackage{tikz}
\RequirePackage{graphicx}
\RequirePackage{pgf}
\RequirePackage{ifthen}
% \RequirePackage[T1]{fontenc}
%\RequirePackage[l2tabu, orthodox]{nag}
\usetikzlibrary{decorations}
\usetikzlibrary{fadings}
\usetikzlibrary{snakes}
\usetikzlibrary{calc}

\RequirePackage{SangerLibrary/SangerResources}
\pagecolor{SangerMedium}
%%%%%%% SANGER SPECIFIC STUFF

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Choose a smaller value for larger fonts
\newcommand{\sangerposter@fontscale}{0.292}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Zoom
%-------------------------------------------------------------------------------
% We scale the page from fontscale * papersize up to papersize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Paper sizes
\newif\if@landscape
\newif\if@geometryKnowsThisSize
\DeclareOptionX{landscape}{\@landscapetrue}
\DeclareOptionX{portrait}{}

\newcommand{\sangerposter@setfinalpapersize}[2]{%
\if@geometryKnowsThisSize
  \setlength{\sangerposter@finalpaperwidth}{#1}%
  \setlength{\sangerposter@finalpaperheight}{#2}%
\else
\if@landscape
% Transpose length, if geometry does not handle the papersize based on the key
  \setlength{\sangerposter@finalpaperwidth}{#2}%
  \setlength{\sangerposter@finalpaperheight}{#1}%
\else
  \setlength{\sangerposter@finalpaperwidth}{#1}%
  \setlength{\sangerposter@finalpaperheight}{#2}%
\fi
\fi
}

% Default paperwidth and paperheight = a0paper
\DeclareOptionX{paperwidth}[841mm]{\setlength{\sangerposter@finalpaperwidth}{#1}}
\DeclareOptionX{paperheight}[1189mm]{\setlength{\sangerposter@finalpaperheight}{#1}}
\DeclareOptionX{archA}         {                           \sangerposter@setfinalpapersize{9in}{12in}}%
\DeclareOptionX{archB}         {                           \sangerposter@setfinalpapersize{12in}{18in}}%
\DeclareOptionX{archC}         {                           \sangerposter@setfinalpapersize{18in}{24in}}%
\DeclareOptionX{archD}         {                           \sangerposter@setfinalpapersize{24in}{36in}}%
\DeclareOptionX{archE}         {                           \sangerposter@setfinalpapersize{36in}{48in}}%
\DeclareOptionX{archE1}        {                           \sangerposter@setfinalpapersize{30in}{42in}}%
\DeclareOptionX{archE2}        {                           \sangerposter@setfinalpapersize{26in}{38in}}%
\DeclareOptionX{archE3}        {                           \sangerposter@setfinalpapersize{27in}{39in}}%
\DeclareOptionX{a0paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{841mm}{1189mm}}%g
\DeclareOptionX{a1paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{594mm}{841mm}}%g
\DeclareOptionX{a2paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{420mm}{594mm}}%g
\DeclareOptionX{a3paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{297mm}{420mm}}%g
\DeclareOptionX{a4paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{210mm}{297mm}}%g
\DeclareOptionX{a5paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{148mm}{210mm}}%g
\DeclareOptionX{a6paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{105mm}{148mm}}%g
\DeclareOptionX{b0paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{1000mm}{1414mm}}%g
\DeclareOptionX{b1paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{707mm}{1000mm}}%g
\DeclareOptionX{b2paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{500mm}{707mm}}%g
\DeclareOptionX{b3paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{353mm}{500mm}}%g
\DeclareOptionX{b4paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{250mm}{353mm}}%g
\DeclareOptionX{b5paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{176mm}{250mm}}%g
\DeclareOptionX{b6paper}       {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{125mm}{176mm}}%g
\DeclareOptionX{ansiapaper}    {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{8.5in}{11in}}%
\DeclareOptionX{ansibpaper}    {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{11in}{17in}}%
\DeclareOptionX{ansicpaper}    {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{17in}{22in}}%
\DeclareOptionX{ansidpaper}    {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{22in}{34in}}%
\DeclareOptionX{ansiepaper}    {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{34in}{44in}}%
\DeclareOptionX{letterpaper}   {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{8.5in}{11in}}%
\DeclareOptionX{legalpaper}    {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{8.5in}{14in}}%
\DeclareOptionX{executivepaper}{\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{7.25in}{10.5in}}%
\DeclareOptionX{screen}        {\@geometryKnowsThisSizetrue\sangerposter@setfinalpapersize{225mm}{180mm}}%g

% Margin
\setlength{\sangerposter@finalmargin}{1.5cm}
\DeclareOptionX{fontscale}[0.292]{\renewcommand{\sangerposter@fontscale}{#1}}
\DeclareOptionX{margin}   [1.5cm]{\setlength{\sangerposter@finalmargin}{#1}}

% move text/poster body to the right (or to the left if negative)
\newlength{\sangerposter@movebody}
\setlength{\sangerposter@movebody}{0cm}
\DeclareOptionX{movebody}[0cm]{\setlength{\sangerposter@movebody}{#1}}


\newif\if@debug
\DeclareOptionX{debug}{\@debugtrue}
%% Will be passed on to other packages (xcolor and geometry), still we don't want unused warnings
\DeclareOptionX{table}{}
\DeclareOptionX{showframe}{}

\ProcessOptionsX

\if@debug
\newcommand{\debug}[1]{\typeout{#1}}
\else
\newcommand{\debug}[1]{}
\fi



\setlength{\sangerposter@basepaperwidth} {\sangerposter@fontscale\sangerposter@finalpaperwidth }
\setlength{\sangerposter@basepaperheight}{\sangerposter@fontscale\sangerposter@finalpaperheight}
\setlength{\sangerposter@basemargin}     {\sangerposter@fontscale\sangerposter@finalmargin}
\newlength{\sangerposter@basemarginright}
\setlength{\sangerposter@basemarginright}{\sangerposter@basemargin}
\addtolength{\sangerposter@basemarginright}{-\sangerposter@fontscale\sangerposter@movebody}
\newlength{\sangerposter@basemarginleft}
\setlength{\sangerposter@basemarginleft}{\sangerposter@basemargin}
\addtolength{\sangerposter@basemarginleft}{\sangerposter@fontscale\sangerposter@movebody}

\typeout{Paperwidth=\the\sangerposter@finalpaperwidth}
\typeout{Paperheight=\the\sangerposter@finalpaperheight}
\typeout{BasePaperwidth=\the\sangerposter@basepaperwidth}
\typeout{BasePaperheight=\the\sangerposter@basepaperheight}
\usepackage[
   paperwidth=\sangerposter@basepaperwidth,
   paperheight=\sangerposter@basepaperheight,
   tmargin=\sangerposter@basemargin,
   bmargin=\sangerposter@basemargin,
   lmargin=\sangerposter@basemarginleft,
   rmargin=\sangerposter@basemarginright,
   ]{geometry} 

\usepackage{pgfpages}
\if@landscape
\if@geometryKnowsThisSize
\pgfpagesuselayout{resize to}[physical paper width=\sangerposter@finalpaperheight,physical paper height=\sangerposter@finalpaperwidth]
\else
\pgfpagesuselayout{resize to}[physical paper width=\sangerposter@finalpaperwidth,physical paper height=\sangerposter@finalpaperheight]
\fi
\else
\pgfpagesuselayout{resize to}[physical paper width=\sangerposter@finalpaperwidth,physical paper height=\sangerposter@finalpaperheight]
\fi



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Default functions for borders/backgrounds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% These functions will all be redefined from the actual option values. In
%% particular, they will be set to \sangerposter@optionname@optionvalue, which
%% should do the actual work / setting for that particular optionvalue.

\newcommand{\sangerposterPosterDrawBackground}[2]{} % Draw poster background

\newcommand{\sangerposterBoxGetShape}{}             % Returns path of text box shape
\newcommand{\sangerposterBoxDrawBackground}[2]{}    % Draw bg of boxes
\newcommand{\sangerposterBoxDrawBorder}[1]{}        % Draw border of individual boxes

\newcommand{\sangerposterHeaderGetShape}{}          % Returns path of text box shape
\newcommand{\sangerposterHeaderSetShade}[3]{}       % Set bg style for box headers
\newcommand{\sangerposterHeaderDrawBackground}[3]{} % Draw background of box header
\newcommand{\sangerposterHeaderDrawBorder}[1]{}     % Draw border of box header
\newcommand{\sangerposterHeaderDrawText}[1]{}       % Draw text inside box header

\newcommand{\@@previousbox}{notset}             % stores the previously processed box for below=auto

% Function to set a user-defined background
\newcommand{\sangerposter@backgroundCmd}{\error{No background command defined. Use \background{...} to define background}}
\newcommand{\background}[1]{\renewcommand{\sangerposter@backgroundCmd}{#1}}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Handle poster and box options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\debug{Handling keys}

%%
%% POSTER OPTIONS
%%
%% Store all poster options in variables of the form \sangerposter@option
%% choose-keys also store the index in \sangerposter@optionnr
%% choose-keys typically also assign a function
\definecolor{sangerposter@silver}{cmyk}{0,0,0,0.7}
\define@boolkey[ba]{poster}[sangerposter@]{grid}                 [false]       {}
\define@boolkey[ba]{poster}[sangerposter@]{eyecatcher}           [true]        {}
\define@cmdkey [ba]{poster}[sangerposter@]{headerheight}         [0.1\textheight]{}
\define@cmdkey [ba]{poster}[sangerposter@]{columns}              [{}]          {}
\define@cmdkey [ba]{poster}[sangerposter@]{colspacing}           [1em]         {}
\define@cmdkey [ba]{poster}[sangerposter@]{bgColorOne}           [sangerposter@silver]{}
\define@cmdkey [ba]{poster}[sangerposter@]{bgColorTwo}           [green]       {}

% background can be one of: shadeLR, shadeTB, plain, user, none
\define@choicekey*+[ba]{poster}{background}%
    [\sangerposter@background\sangerposter@backgroundnr]%
    {shadeLR, shadeTB, plain, user, none} [plain] {%
  \debug{Poster background: \sangerposter@background}
  \renewcommand{\sangerposterPosterDrawBackground}[2]{
      \csname sangerposter@background@\sangerposter@background\endcsname{##1}{##2}}
}{
  \PackageWarning{sangerposter}{Unknown background `\sangerposter@background' (use
      shadeLR, shadeTB, plain, none, or user). If user is used, you also
      have to define \background{...}.}
  \renewcommand{\sangerposterPosterDrawBackground}[2]{\sangerposter@background@none{##1}{##2}}
}


%%
%%  BOX OPTIONS
%%
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{cornerradius}          [1em]         {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{boxheaderheight}       [2em]         {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{boxpadding}            [0.5em]       {}


% textborder can be one of: none, bars, coils, triangles, rectangle, rounded,
% roundedleft, roundedsmall, faded; UNIMPLEMENTED: roundedright
\edef\sangerposter@box@textborder@validvalues{none,bars,coils,triangles,rectangle,rounded,roundedleft,roundedsmall,faded}
\define@choicekey*+[ba]{posterbox}{textborder}%
    [\sangerposter@box@textborder\sangerposter@box@textbordernr]%
    {none,bars,coils,triangles,rectangle,rounded,roundedleft,roundedright,roundedsmall,faded} [rounded] {%
  \debug{Text border: \sangerposter@box@textborder}
  \renewcommand{\sangerposterBoxGetShape}{
      \csname sangerposter@box@boxshape@\sangerposter@box@textborder\endcsname}
  \renewcommand{\sangerposterBoxDrawBorder}[1]{
      \csname sangerposter@box@drawborder@\sangerposter@box@textborder\endcsname{##1}}
}{
  \PackageWarning{sangerposter}{Unknown text-border style `\sangerposter@box@textborder'.
      Edit your file to choose a valid option (\sangerposter@box@textborder@validvalues).}
  \renewcommand{\sangerposterBoxGetShape}{\sangerposter@boxshape@rectangle}
  \renewcommand{\sangerposterBoxDrawBorder}[1]{\sangerposter@drawborder@rectangle{##1}}
}

% boxshade can be one of: shadeLR, shadeTB, plain, none
\define@choicekey*+[ba]{posterbox}{boxshade}%
    [\sangerposter@box@boxshade\sangerposter@box@boxshadenr]%
    {shadelr,shadetb,plain,none} [none] {%
  \debug{Box shade: \sangerposter@box@boxshade}
  \renewcommand{\sangerposterBoxDrawBackground}[2]{
      \csname sangerposter@box@drawbackground@\sangerposter@box@boxshade\endcsname{##1}{##2}}
}{
  \PackageWarning{sangerposter}{Unknown boxshade style `\sangerposter@boxshade'.
      Edit your file to choose a valid option.}
  \renewcommand{\sangerposterBoxDrawBackground}[2]{\sangerposter@box@drawbackground@none{##1}{##2}}
}

% headershade can be one of: shade-lr, shade-tb, shade-tb-inverse, plain
\define@choicekey*+[ba]{posterbox}{headershade}%
    [\sangerposter@box@headershade\sangerposter@box@headershadenr]%
    {shadelr,shadetb,shadetbinverse,plain} [shadelr] {%
  \debug{Header shade: \sangerposter@box@headershade}
  \renewcommand{\sangerposterHeaderSetShade}[3]{
      \csname sangerposter@box@headershade@\sangerposter@box@headershade\endcsname{##1}{##2}{##3}}
}{
  \PackageWarning{sangerposter}{Unknown headershade style `\sangerposter@box@headershade'.
      Edit your file to choose a valid option.}
  \renewcommand{\sangerposterHeaderSetShade}[3]{\sangerposter@box@headershade@none{##1}{##2}{##3}}
}

% headershape can be one of: rectangle, rounded, smallrounded, roundedleft, roundedright
\define@choicekey*+[ba]{posterbox}{headershape}%
    [\sangerposter@box@headershape\sangerposter@box@headershapenr]%
    {rectangle,rounded,smallrounded,roundedleft,roundedright} [roundedright] {%
  \debug{Header shape: \sangerposter@box@headershape}
  \renewcommand{\sangerposterHeaderGetShape}{
      \csname sangerposter@box@headershape@\sangerposter@box@headershape\endcsname}
  \renewcommand{\sangerposterHeaderDrawText}[1]{
      \csname sangerposter@box@headerdrawtext@\sangerposter@box@headershape\endcsname{##1}}
  \renewcommand{\sangerposterHeaderDrawBorder}[1]{
      \csname sangerposter@box@headerdrawborder@\sangerposter@box@headershape\endcsname{##1}}
}{
  \PackageWarning{sangerposter}{Unknown headershape style `\sangerposter@headershape'.
      Edit your file to choose a valid option.}
  \renewcommand{\sangerposterHeaderGetShape}{\sangerposter@box@headershape@rectangle}
  \renewcommand{\sangerposterHeaderDrawText}[1]{\sangerposter@box@headerdrawtext@rectangle{##1}}
  \renewcommand{\sangerposterHeaderDrawBorder}[1]{\sangerposter@box@headerdrawborder@rectangle{##1}}
}

% headerborder can be one of: open, closed, none
\define@choicekey*+[ba]{posterbox}{headerborder}%
    [\sangerposter@box@headerborder\sangerposter@box@headerbordernr]%
    {open,closed,none} [open] {%
  \debug{Header border: \sangerposter@box@headerborder}
%   \renewcommand{\sangerposterHeaderBorder}{
%       \csname sangerposter@headerborder@\sangerposter@box@headerborder\endcsname}
}{
  \PackageWarning{sangerposter}{Unknown headerborder style `\sangerposter@headerborder'.
      Edit your file to choose a valid option.}
%   \renewcommand{\sangerposterHeaderBorder}{\sangerposter@box@headerborder@rectangle}
}


\define@cmdkey[ba]{posterbox}[sangerposter@box@]{borderColor}           [SangerDark]      {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{headerColorOne}        [SangerDeep]         {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{headerColorTwo}        [SangerDeep]       {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{headerFontColor}       [white]       {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{boxColorOne}           [white]     {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{boxColorTwo}           [white]        {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{headerfont}            [\Large]   {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{textfont}              []          {}

\define@cmdkey[ba]{posterbox}[sangerposter@box@]{linewidth}             [2pt]         {}

\define@cmdkey[ba]{posterbox}[sangerposter@box@]{below}  [notset]{}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{above}  [notset]{}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{aligned}[notset]{}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{bottomaligned}[notset]{}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{column} [0]     {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{row}    [0]     {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{span}   [1]     {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{height} [auto]  {}
\define@cmdkey[ba]{posterbox}[sangerposter@box@]{name}   [noname]{}

% Set some default values, the poster and posterbox environments can override:
\setkeys[ba]{poster}{
  % Debug grid
  grid=false,
  % Is there an eyecatcher image
  eyecatcher=false,
  columns={},
  % Colours
  bgColorOne=SangerMedium,
  bgColorTwo=SangerMedium,
  %
  colspacing=1em,
  headerheight=0.1\textheight,
  background=shadelr,
}{}
\setkeys[ba]{posterbox}{
  % Position
  column=0,row=0,span=1,
  below=notset,above=notset,
  bottomaligned=notset,
  aligned=notset,
  height=auto,
  % Name
  name=noname,
  % Box design: border:
  linewidth=2pt,
  borderColor=SangerDark,
  cornerradius=1em,
  % text box:
  textfont={\Helvetica},
  boxshade=plain,
  boxColorOne=white,
  boxColorTwo=white,
  textborder=rounded,
  boxpadding=0.5em,
  % header
  headerfont=\Large\Wellcome,% or headerfont=\color{white}\textsf\textbf
  headerFontColor=white,
  headerColorOne=SangerDeep,
  headerColorTwo=SangerDeep,
  headershape=rounded,
  headershade=plain,
  headerborder=open,
  boxheaderheight=2em,
}{}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Background options and functions (one function for each possible value)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\sangerposter@background@shadelr}[2]{
  \debug{sangerposter: Using shade left right background.}
  \begin{tikzpicture}[remember picture,overlay]%
    \shade [shading=axis,left color=#1,right color=#2] (current page.north west)
           rectangle(current page.south east);
  \end{tikzpicture}%
}
\newcommand{\sangerposter@background@shadetb}[2]{
  \debug{sangerposter: Using shade top to bottom background.}
  \begin{tikzpicture}[remember picture,overlay]%
    \shade [shading=axis,top color=#1,bottom color=#2] (current page.north west)
           rectangle(current page.south east);
  \end{tikzpicture}%
}
\newcommand{\sangerposter@background@plain}[2]{
  \debug{sangerposter: Using plain background.}
  \begin{tikzpicture}[remember picture,overlay]%
    \fill [fill=#1] (current page.north west) rectangle(current page.south east);
  \end{tikzpicture}%
}
\newcommand{\sangerposter@background@user}[2]{
  \debug{sangerposter: Using user background.}
  \sangerposter@backgroundCmd%
}
\newcommand{\sangerposter@background@none}[2]{
  \debug{sangerposter: Using no background.}
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Return shape path of text box (depending on the box shape)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\sangerposter@box@boxshape@none}{\sangerposter@box@boxshape@rectangle}
\newcommand{\sangerposter@box@boxshape@bars}{
  (\sangerposter@box@name tnw) -- (\sangerposter@box@name sw) %
  (\sangerposter@box@name se) -- (\sangerposter@box@name tne)
}
\newcommand{\sangerposter@box@boxshape@coils}{\sangerposter@box@boxshape@bars}
\newcommand{\sangerposter@box@boxshape@triangles}{\sangerposter@box@boxshape@bars}
\newcommand{\sangerposter@box@boxshape@rectangle}{
  (\sangerposter@box@name tnw) -- (\sangerposter@box@name sw) -- %
  (\sangerposter@box@name se) -- (\sangerposter@box@name tne)%
}
\newcommand{\sangerposter@box@boxshape@faded}{
  (\sangerposter@box@name tnw) -- (\sangerposter@box@name sw) %
  (\sangerposter@box@name tne) -- (\sangerposter@box@name se)
  }
\newcommand{\sangerposter@box@boxshape@rounded}{
  [rc] \sangerposter@box@boxshape@rectangle%
}
\newcommand{\sangerposter@box@boxshape@roundedsmall}{
  [src] \sangerposter@box@boxshape@rectangle
}
\newcommand{\sangerposter@box@boxshape@roundedleft}{
  (\sangerposter@box@name tnw) {[rc]-- (\sangerposter@box@name sw)} -- %
  (\sangerposter@box@name se) -- (\sangerposter@box@name tne)%
}
\newcommand{\sangerposter@box@boxshape@roundedright}{
  (\sangerposter@box@name tnw) -- (\sangerposter@box@name sw) {[rc]-- %
  (\sangerposter@box@name se)} -- (\sangerposter@box@name tne)%
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Draw box background (one function for each possible value)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% These functions take no arguments
\newcommand{\sangerposter@box@drawbackground@none}[2]{
  \tikzstyle{box colors}=[]
}
\newcommand{\sangerposter@box@drawbackground@plain}[2]{
  \tikzstyle{box colors}=[fill=#1]
  \fill[box colors] \sangerposterBoxGetShape;
}
\newcommand{\sangerposter@box@drawbackground@shadelr}[2]{
  \tikzstyle{box colors}=[shading=axis, left color=#1, right color=#2]%
  \fill[box colors] \sangerposterBoxGetShape;
}
\newcommand{\sangerposter@box@drawbackground@shadetb}[2]{
  \tikzstyle{box colors}=[shading=axis, top color=#1, bottom color=#2]%
  \fill[box colors] \sangerposterBoxGetShape;
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Draw box border
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% These functions take two arguments: borderColor
\newcommand{\sangerposter@box@drawborder@none}[1]{}
\newcommand{\sangerposter@box@drawborder@bars}[1]{
  \draw[color=#1] \sangerposterBoxGetShape;%
}
\newcommand{\sangerposter@box@drawborder@coils}[1]{
  \draw[color=#1,segment amplitude=0.35em,segment length=0.4em,snake=coil] \sangerposterBoxGetShape;%
}
\newcommand{\sangerposter@box@drawborder@triangles}[1]{
  \draw[color=#1,segment amplitude=0.2em,segment length=0.4em,snake=triangles] \sangerposterBoxGetShape;%
}
\newcommand{\sangerposter@box@drawborder@rectangle}[1]{
  \draw[color=#1] \sangerposterBoxGetShape;%
}
\newcommand{\sangerposter@box@drawborder@rounded}[1]{
  \draw[color=#1] \sangerposterBoxGetShape;%
}
\newcommand{\sangerposter@box@drawborder@roundedleft}[1]{
  \draw[color=#1] \sangerposterBoxGetShape;%
}
\newcommand{\sangerposter@box@drawborder@roundedright}[1]{
  \draw[color=#1] \sangerposterBoxGetShape;%
}
\newcommand{\sangerposter@box@drawborder@faded}[1]{
  % This is the right way to do it, but it does not work with evince, and has problems during printing, so instead we do
  %\draw[color=#1,path fading=south] \sangerposterBoxGetShape;%
  % this
 %\foreach \x in {0,1,...,90} { \draw[color=#1!\x] ($(\sangerposter@box@name tnw)!{(100-\x)/100}!(\sangerposter@box@name sw)$) -- ($(\sangerposter@box@name tnw)!{(100-(\x+10))/100}!(\sangerposter@box@name sw)$); }% 
 %\foreach \x in {0,1,...,90} { \draw[color=#1!\x] ($(\sangerposter@box@name tne)!{(100-\x)/100}!(\sangerposter@box@name se)$) -- ($(\sangerposter@box@name tne)!{(100-(\x+10))/100}!(\sangerposter@box@name se)$); }%
  \foreach \x in {0,1,...,99} { \draw[color=#1,opacity=\x/100] ($(\sangerposter@box@name tnw)!{(100-\x)/100}!(\sangerposter@box@name sw)$) -- ($(\sangerposter@box@name tnw)!{(100-(\x+1))/100}!(\sangerposter@box@name sw)$); }% 
  \foreach \x in {0,1,...,99} { \draw[color=#1,opacity=\x/100] ($(\sangerposter@box@name tne)!{(100-\x)/100}!(\sangerposter@box@name se)$) -- ($(\sangerposter@box@name tne)!{(100-(\x+1))/100}!(\sangerposter@box@name se)$); }%
}
\newcommand{\sangerposter@box@drawborder@roundedsmall}[1]{
  \draw[color=#1] \sangerposterBoxGetShape;%
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Return shape path of text box (depending on the box shape)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% These functions take no arguments
% TODO: For headerborder==none, use (\sangerposter@box@name outer tnw) instead!
\newcommand{\sangerposter@box@headershape@rectangle}{%
  (\sangerposter@box@name tnw) -- (\sangerposter@box@name nw) -- %
  (\sangerposter@box@name ne) -- (\sangerposter@box@name tne)%
}
\newcommand{\sangerposter@box@headershape@smallrounded}{%
  (\sangerposter@box@name tnw) {[src] -- (\sangerposter@box@name nw) -- %
  (\sangerposter@box@name ne)} -- (\sangerposter@box@name tne)%
}
\newcommand{\sangerposter@box@headershape@roundedright}{%
  (\sangerposter@box@name tnw) -- (\sangerposter@box@name nw) {[rc] -- %
  (\sangerposter@box@name ne)} -- (\sangerposter@box@name tne)%
}
\newcommand{\sangerposter@box@headershape@roundedleft}{%
  (\sangerposter@box@name tnw) {[rc]-- (\sangerposter@box@name nw)} -- %
  (\sangerposter@box@name ne) -- (\sangerposter@box@name tne)%
}
\newcommand{\sangerposter@box@headershape@rounded}{%
  (\sangerposter@box@name tnw) {[rc] -- (\sangerposter@box@name nw) -- %
  (\sangerposter@box@name ne) } -- (\sangerposter@box@name tne)%
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Header text drawing (one function for each possible value of headershape)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% These functions take one argument: the header text
\newcommand{\sangerposter@box@headerdrawtext@rectangle}[1]{
  \path (\sangerposter@box@name nw) +(0em,-0.5\sangerposter@box@@boxheaderheight) node[anchor=west,inner sep=0.4em] {#1};%
}
\newcommand{\sangerposter@box@headerdrawtext@smallrounded}[1]{
  \path (\sangerposter@box@name nw) +(0.5\boxwidth,-0.5\sangerposter@box@@boxheaderheight) node[anchor=center] {#1};%
}
\newcommand{\sangerposter@box@headerdrawtext@roundedright}[1]{
  \path (\sangerposter@box@name nw) +(0em,-0.5\sangerposter@box@@boxheaderheight)%
        node[anchor=west,inner sep=0.4em,text depth=0.0em] {#1};%
}
\newcommand{\sangerposter@box@headerdrawtext@roundedleft}[1]{
  \path (\sangerposter@box@name nw) +(0em,-0.5\sangerposter@box@@boxheaderheight)%
        node[anchor=west,inner sep=0.4em] {#1};%
}
\newcommand{\sangerposter@box@headerdrawtext@rounded}[1]{
    \path (\sangerposter@box@name nw) +(0.5\boxwidth,-0.5\sangerposter@box@@boxheaderheight) node[anchor=center] {#1};%
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Header shade options and functions (one function for each possible value)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% These functions take two arguments: headerColorOne, headerColorTwo and borderColor
\newcommand{\sangerposter@box@headershade@shadelr}[3]{
    \debug{Header-Shade: Shade Left - Right}
    \tikzstyle{header colors}=[%
      color=#3,%
      shading=axis,%
      left color=#1,%
      right color=#2%
    ]%
}
\newcommand{\sangerposter@box@headershade@shadetb}[3]{
    \debug{Header-Shade: Shade Top - Bottom}
    \tikzstyle{header colors}=[%
      color=#3,%
      shading=axis,%
      top color=#1,%
      bottom color=#2%
    ]%
}
\newcommand{\sangerposter@box@headershade@shadetbinverse}[3]{
    \tikzstyle{header colors}=[%
      top color=#1!75!#2,%
      bottom color=#2!100!#1,%
      shading angle=20%
    ]%
    \colorlet{sangerposterHeaderFontColor}{white}%
}
\newcommand{\sangerposter@box@headershade@plain}[3]{
    \debug{Header-Shade: Plain}
    \tikzstyle{header colors}=[%
      color=#3,%
      fill=#1%
    ]%
}
\newcommand{\sangerposter@box@headershade@none}[3]{
    \debug{Header-Shade: none}
    \tikzstyle{header colors}=[]
}




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The main poster environment
%%% \begin{sangerposter}{settings}{Eye Catcher}{Title}{Author}{University Logo}
%%%-----------------------------------------------------------------------------
%%% The settings are
%%%   - grid=true,[false]:Show grid to help with alignment
%%%   - colspacing=0.7em: Column spacing
%%%   - columns=4:        number of columns (default 4 in landscape and 3 in portrait format) (maximum number is 6)
%%%   - color=[orange]:   xcolor color definition used as the main color of the poster
%%%   - colortwo=[white]: The other color for gradient based layouts
%%%   - textborder=none,bars,coils,triangles,rectangle,rounded,roundedsmall,roundedleft,roundedright,[faded]
%%%                       The style of the box around the text area
%%%   - headerborder=none,closed,open
%%%                       No extra border around box header, full border around box header or border that is open below.
%%%   - headershape=rectangle,rounded,roundedleft,roundedright
%%%                       Shape of the box-header region
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{poster}[5]
{%
  \thispagestyle{empty}% Suppress Page Number
  \debug{Poster Starts}%
  % This setkeys call parses all provided options and depending on the option
  % value, assigns different handler functions to the \sangerposter(Box|Header)*
  % functions. Once that is done, we don't have to care about particular
  % values for border, shading, etc. All we have to do is call the
  % handler functions and let them do their job.
  % This also allows the user to override the poster-wide defaults on a per-box
  % basis.
  \setkeys[ba]{posterbox,poster}{#1}%
  %
  % TODO: Move all those assignments to the key macros!
  % Parse Keys%
  \colorlet{bgColorOne}{\sangerposter@bgColorOne}
  \colorlet{bgColorTwo}{\sangerposter@bgColorTwo}
  %
  %% Boxes%
  \setlength{\headerheight}{\sangerposter@headerheight}%
  \setlength{\colheight}{\textheight-\sangerposter@headerheight}%
  \renewcommand{\@@previousbox}{notset}

  \debug{Format}%
  % Set default for columns if unset (4 for landscape, 3 for portrait)
  \ifthenelse{\equal{\sangerposter@columns}{}}{%
    \if@landscape
      \renewcommand{\sangerposter@columns}{4}
    \else
      \renewcommand{\sangerposter@columns}{3}
    \fi
    %
  }{}
  %
  \debug{Columns: \sangerposter@columns}%
  \setlength{\sangerposter@@colspacing}{\sangerposter@colspacing}%
  \setlength{\colwidth}{\textwidth}%
  \addtolength{\colwidth}{\sangerposter@@colspacing*(1-\sangerposter@columns)}%
  \ifcase\sangerposter@columns\relax
    \error{You need to have at least one column!}
  \or % 1
    \setlength{\colwidth}{\colwidth}%
  \or % 2
    \setlength{\colwidth}{0.5\colwidth}%
  \or % 3
    \setlength{\colwidth}{0.3333333333333\colwidth}%
  \or % 4
    \setlength{\colwidth}{0.25\colwidth}%
  \or % 5
    \setlength{\colwidth}{0.2\colwidth}%
  \or % 6
    \setlength{\colwidth}{0.16666666666\colwidth}%
  \else % >6
    \error{You do not want so many columns}
  \fi
  %
  \newcommand{\sangerposter@reference}{north west}%
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % A box with a header and some content. The basic unit of the poster%
  %---------------------------------------------------------------------------%
  % Each box has a name and can be placed absolutely or relatively.%
  % The only inconvenience is that you can only specify a relative position %
  % towards an already declared box. So if you have a box attached to the %
  % bottom, one to the top and a third one which should be inbetween, you %
  % have to specify the top and bottom boxes before you specify the middle %
  % box.%
  %%
  % below=  name of other node%
  % above=  name of other node%
  % aligned=name of other node%
  % bottomaligned=name of other node%
  % column= [0]     %
  % row=    [0]     %
  % span=   [1]     %
  % height= <size in percent of column height>,[auto]%
  % name=   [noname]%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Backward-compatibility definition (\headerbox command uses posterbox env):
  \newcommand{\headerbox}[3]{%
    \begin{posterbox}[##2]{##1}
      ##3
    \end{posterbox}
  }

  \newenvironment{posterbox}[2][]{%
    \debug{Poster box options: ##1}%
    % Override the poster-wide defaults on a per-box basis
    \setkeys[ba]{posterbox}{##1}%
  %
    \def\sangerposter@box@title{##2}
      %
    \colorlet{borderColor}{\sangerposter@box@borderColor}
    \colorlet{headerColorOne}{\sangerposter@box@headerColorOne}
    \colorlet{headerColorTwo}{\sangerposter@box@headerColorTwo}
    \colorlet{headerFontColor}{\sangerposter@box@headerFontColor}
    \colorlet{boxColorOne}{\sangerposter@box@boxColorOne}
    \colorlet{boxColorTwo}{\sangerposter@box@boxColorTwo}
  %
    \setlength{\sangerposter@box@@cornerradius}{\sangerposter@box@cornerradius}%
    \ifthenelse{\equal{\sangerposter@box@title}{}}{% No header
    \setlength{\sangerposter@box@@boxheaderheight}{0pt}%
    }{ % With header
    \setlength{\sangerposter@box@@boxheaderheight}{\sangerposter@box@boxheaderheight}%
    }%
    \setlength{\sangerposter@box@@boxpadding}{\sangerposter@box@boxpadding}%


    %% The columns is always given absolute
    % boxstartx = column * colwidth + column * colspacing
    \setlength{\boxstartx}{(\colwidth+\sangerposter@@colspacing)*\sangerposter@box@column}%
  %
    %% The width is gvien absolute
    % Box Width = span * colwidth + (span-1) * colspacing
    \setlength{\boxwidth}{\sangerposter@box@span\colwidth} %
    \addtolength{\boxwidth}{\sangerposter@@colspacing*(\sangerposter@box@span-1)}%
  %
    %% Measure the content of the box%
    \setbox\sangerposter@box@content=\hbox\bgroup%
      \begin{pgfinterruptpicture}%
        \begin{minipage}[t]{\boxwidth-\sangerposter@box@@boxpadding*2}%
          \sangerposter@box@textfont\bgroup%
  }% End of posterbox preamble
  %%% HERE COME THE ACTUAL CONTENTS OF THE HEADERBOX ENVIRONMENT
  {% posterbox handling after contents (i.e. drawing everything)
          \egroup%
        \end{minipage}%
      \end{pgfinterruptpicture}%
    \egroup%
    \setlength{\boxheight}{\ht\sangerposter@box@content}%
    \addtolength{\boxheight}{\dp\sangerposter@box@content}%
    \addtolength{\boxheight}{\sangerposter@box@@boxheaderheight} % Header%
    \addtolength{\boxheight}{2\sangerposter@box@@boxpadding} % Inner Sep
  %
    \ifthenelse{\equal{\sangerposter@box@height}{bottom}}{%
    }{\ifthenelse{\equal{\sangerposter@box@height}{auto}}{%
    }{ % Neither auto nor bottom%
      \setlength{\boxheight}{\sangerposter@box@height\colheight}%
    }}%
  %
    %% Determine the box position%
    \debug{Setting Coordinates}%
    \debug{Upper Right}%
    \debug{\sangerposter@box@name}%
  %
    %%% Upper Right Corner%
    % if below=auto, set it to the previous box
    % TODO: We should generalize this to the previous box of the used column,
    %       currently we use the previous box, which might be in a different column
    \ifthenelse{\equal{\sangerposter@box@below}{auto}}{%
        \edef\sangerposter@box@below{\@@previousbox}
        \debug{Box \sangerposter@box@name  has below=auto, placing it below box \sangerposter@box@below.}
    }{}
    \xdef\@@previousbox{\sangerposter@box@name}

    \ifthenelse{\not\equal{\sangerposter@box@below}{notset} }{%
      %% Below%
      \debug{Below}%
      \path[shape=coordinate] (\boxstartx,0pt |- \sangerposter@box@below se) ++(0pt,-\sangerposter@@colspacing) coordinate(\sangerposter@box@name nw);%
    }{%
      \ifthenelse{\not\equal{\sangerposter@box@aligned}{notset} }{%
        %% Aligned%
        \debug{Aligned: \sangerposter@box@aligned}%
        \path[shape=coordinate] (\boxstartx,0pt |- \sangerposter@box@aligned nw)                           coordinate(\sangerposter@box@name nw);%
      }{%
        %% Fixed%
        \debug{Fixed}%
        \setlength{\boxstarty}{\sangerposter@box@row\colheight}%
        \path[shape=coordinate] (\boxstartx,\colheight-\boxstarty)                                                  coordinate(\sangerposter@box@name nw);%
    }}%
  %
    %% Lower Left Corner%
    \debug{Lower Left}%
    \ifthenelse{\equal{\sangerposter@box@above}{bottom}}{%
      %% Above = Bottom%
      \debug{Above bottom}%
      \ifthenelse{\equal{\sangerposter@box@below}{notset} \and \equal{\sangerposter@box@aligned}{notset}}{%
      \path[shape=coordinate] (\boxstartx,\boxheight)                                                              coordinate(\sangerposter@box@name nw);%
      }{}%
      \path[shape=coordinate] (\boxstartx+\boxwidth,0pt)                                                           coordinate(\sangerposter@box@name se);%
      }{\ifthenelse{\not \equal{\sangerposter@box@bottomaligned}{notset}}{%
        \path[shape=coordinate] (\boxstartx+\boxwidth,0pt |- \sangerposter@box@bottomaligned se)                       coordinate(\sangerposter@box@name se);%
      }{{\ifthenelse{\not \equal{\sangerposter@box@above}{notset}}{%
        %% Above = Node%
        \path[shape=coordinate] (\boxstartx+\boxwidth,0pt |- \sangerposter@box@above nw)  +(0pt,\sangerposter@@colspacing) coordinate(\sangerposter@box@name se);%
      }{%
        %% Above = notset%
        \debug{Above=not set}%
        \ifthenelse{\equal{\sangerposter@box@height}{bottom}}{%
          %% height=bottom%
          \debug{height=bottom}%
          \path[shape=coordinate] (\boxstartx+\boxwidth,0pt)                                                       coordinate(\sangerposter@box@name se);%
        }{ %% height=auto or fixed%
          \debug{height=auto or fixed}%
          \path[shape=coordinate] (\sangerposter@box@name nw) ++(\boxwidth,-\boxheight)                                coordinate(\sangerposter@box@name se);%
	  }}}}}%
  %
        %
    % Set coordinates relative to nw,se%
    \debug{Fixing Coordinates}%
    \path[shape=coordinate]%
      (\sangerposter@box@name nw) +(0pt,-\sangerposter@box@@boxheaderheight)                coordinate(\sangerposter@box@name tnw)%
      (\sangerposter@box@name nw |- \sangerposter@box@name se)   coordinate(\sangerposter@box@name sw)%
      (\sangerposter@box@name se |- \sangerposter@box@name nw)   coordinate(\sangerposter@box@name ne)%
      (\sangerposter@box@name ne) +(0pt,-\sangerposter@box@@boxheaderheight)                coordinate(\sangerposter@box@name tne)%
%
      (\sangerposter@box@name nw)  +(-0.025em,0pt)           coordinate(\sangerposter@box@name outer nw)%
      (\sangerposter@box@name tnw) +(-0.025em,0pt)           coordinate(\sangerposter@box@name outer tnw)%
      (\sangerposter@box@name sw)  +(-0.025em,0pt)           coordinate(\sangerposter@box@name outer sw)%
  %
      (\sangerposter@box@name ne)  +( 0.025em,0pt)           coordinate(\sangerposter@box@name outer ne)%
      (\sangerposter@box@name tne) +( 0.025em,0pt)           coordinate(\sangerposter@box@name outer tne)%
      (\sangerposter@box@name se)  +( 0.025em,0pt)           coordinate(\sangerposter@box@name outer se);%
  %
      %% Setting the bg colors of the box header
      \sangerposterHeaderSetShade{headerColorOne}{headerColorTwo}{borderColor}
  %
      \tikzstyle{rc}=[rounded corners=\sangerposter@box@@cornerradius];%
      \tikzstyle{src}=[rounded corners=0.5em];%
  %

    %% Now that everything is set up, draw the actual box, with bg and header
    \begin{scope}[line width=\sangerposter@box@linewidth]
      %% Header%
      \debug{Header}%
      \debug{Header-Shape: \sangerposter@box@headershape, header-border: \sangerposter@box@headerborder (\sangerposter@box@headerbordernr)}%
      % TODO: Also turn this last ifcase construct into a handler function
      %       We only need to determine (fill|shade)(draw|)...
  %       \sangerposterHeaderDrawBackground{bgColorOne}{bgColorTwo}{borderColor}
  %       \sangerposterHeaderDrawBorder{borderColor}
    \ifthenelse{\equal{\sangerposter@box@title}{}}{% No header
    }{% With header
      \ifcase\sangerposter@box@headerbordernr\relax%
        % open
        \ifthenelse{\equal{\sangerposter@box@headershade}{plain}}{
          \filldraw  [style=header colors] \sangerposterHeaderGetShape;%
        }{
          \shadedraw [style=header colors] \sangerposterHeaderGetShape;%
        }
      \or
        % closed
        \ifthenelse{\equal{\sangerposter@box@headershade}{plain}}{
          \filldraw  [style=header colors] \sangerposterHeaderGetShape -- cycle;%
        }{
          \shadedraw [style=header colors] \sangerposterHeaderGetShape -- cycle;%
        }
      \or
        % none
        \ifthenelse{\equal{\sangerposter@box@headershade}{plain}}{
          \fill      [style=header colors] \sangerposterHeaderGetShape;%
        }{
          \shade     [style=header colors] \sangerposterHeaderGetShape;%
        }
      \fi
      %
      %% Draw the text inside the box header:
      \sangerposterHeaderDrawText{\color{headerFontColor}\sangerposter@box@headerfont{\sangerposter@box@title}};%
    }
      %
      %% Text borders (border around boxes)
      \debug{Poster boxes}%
      % First set box shade
      \sangerposterBoxDrawBackground{boxColorOne}{boxColorTwo}
      \sangerposterBoxDrawBorder{borderColor}
      %%
      %% Text Box%
      \debug{Drawing Text}%
      \path (\sangerposter@box@name tnw) node(text) [anchor=north west,
            outer sep=-0.000em,text width=\boxwidth-2\sangerposter@box@@boxpadding,inner sep=\sangerposter@box@@boxpadding,
            text justified] {\usebox{\sangerposter@box@content}};%
    \end{scope}
    %
    % Finally store the box name as the previous box for the next call
%     \xdef\@@previousbox{\sangerposter@box@name}%
  }% END of posterbox definition
%
  %% Poster Background%
  \sangerposterPosterDrawBackground{bgColorOne}{bgColorTwo}%
  %% Poster header/title
  \hspace{-1.5em}%
  \begin{tikzpicture}[inner sep=0pt,outer sep=0pt,line width=0.05em]%
    \useasboundingbox (0em,0em) rectangle(\textwidth,\textheight);%
    \path[shape=coordinate]%
      (0pt,\colheight) coordinate(north west) (\textwidth,\colheight) coordinate(north east)%
      (0pt,0pt) coordinate(south west)        (\textwidth,0pt) coordinate(south east);%
%
    \ifsangerposter@eyecatcher% Has eye catcher
      \debug{Eyecatcher found!}
      \setbox\sangerposter@titleimage@left=\hbox{#2}%
    \else% Has no eye catcher%
      \setbox\sangerposter@titleimage@left=\hbox{}%
    \fi%
    \setlength{\sangerposter@titleimage@left@width}{\wd\sangerposter@titleimage@left}%
    \setbox\sangerposter@titleimage@right=\hbox{#5}%
    \setlength{\sangerposter@titleimage@right@width}{\wd\sangerposter@titleimage@right}%
    \setlength{\sangerposter@titleimage@textwidth}{\textwidth}%
    \addtolength{\sangerposter@titleimage@textwidth}{-\sangerposter@titleimage@left@width}%
    \addtolength{\sangerposter@titleimage@textwidth}{-\sangerposter@titleimage@right@width}%

    \debug{#3}
    %
    %
    %      % Draw Header%
    \draw (north west) +(0em,1em+0.5\headerheight) node(image)[anchor=west]   { {\usebox{\sangerposter@titleimage@left }} };%
    \draw (north east) +(0em,1em+0.5\headerheight) node(logo) [anchor=east]   { {\usebox{\sangerposter@titleimage@right}} };%
    %
    \ifsangerposter@eyecatcher% Has eye catcher%
      \draw (image.east) node(title)[anchor=west,text width=\sangerposter@titleimage@textwidth]{%
        \begin{minipage}{\sangerposter@titleimage@textwidth}%
          \begin{center}%
          \textbf{\Huge \Wellcome #3}\\%
          {\Large \Helvetica #4}%
          \end{center}%
        \end{minipage}
      };%
    \else% Has no eye catcher
      \draw (image.east) node(title)[anchor=west]  { {\begin{minipage}{\sangerposter@titleimage@textwidth}{\parbox[t]{0.99\linewidth}{\Wellcome \fontsize{23}{15}\selectfont \raggedright #3}}\\{\Large\Helvetica  #4}\end{minipage}} };%
    \fi
}% END poster begin
{% BEGIN poster end
    % The end, draw gridlines if neccesary
    \ifsangerposter@grid
      \newdimen{\gridpos}
      \typeout{\sangerposter@columns-1}
      \pgfmathsetmacro{\z}{\sangerposter@columns-1}
     \foreach \y in {0,...,\z}
     {
       \setlength{\gridpos}{\y\colwidth+\y\sangerposter@@colspacing}
       \draw[draw=green,draw opacity=0.7] (\gridpos,0pt)  -- (\gridpos,\colheight)
           (\gridpos+\colwidth,0pt)  -- (\gridpos+\colwidth,\colheight);%
     }
     % Horizontal lines, every 0.1:
     %% Explicitly list all percentages, because with {0.0, 0.1, ..., 1.0} we
     %% get rounding errors in the displayed numbers!
     \foreach \y in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
       \draw[draw=green,draw opacity=0.7] (0pt,\colheight-\y\colheight)  --
           (\textwidth,\colheight-\y\colheight) node[anchor=west] {\y};%
    \fi%
  \end{tikzpicture}%
 % \xkvview{}
 \par
}% END poster end




