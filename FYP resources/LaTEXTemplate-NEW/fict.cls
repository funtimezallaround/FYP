\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{fict}[2022/12/20 Faculty of ICT Dissertation template]

% Load the book class.
\LoadClass[12pt, a4paper, oneside, openany]{book}

% Use OTF fonts and set Lato as the default font.
\RequirePackage{fontspec}
\RequirePackage[default]{lato}

% Set the margins of the document.
\RequirePackage[a4paper, top=2.54cm, bottom=2.54cm, left=3cm, right=2cm]{geometry}

% Set the text to be left-aligned and indent of 1.27cm
\raggedright{}
\setlength{\parindent}{1.27cm}

% Set 1.5 line spacing.
\RequirePackage{setspace}
\onehalfspacing{}

% Enable figure insertion.
\RequirePackage{graphicx}

% Generate hyperlinks.
\RequirePackage[hidelinks,pdfusetitle]{hyperref}

% Math symbols and access to the '\eqref' command.
\RequirePackage{amsmath}

% Reference style and engine.
\RequirePackage[bibstyle=ieee,citestyle=numeric-comp]{biblatex}

% Table configuration.
\RequirePackage{tabularray}
%% For aligning numbers in a table with the period symbol.
\RequirePackage{siunitx}
\sisetup{detect-all}
\UseTblrLibrary{siunitx}

% Chapter page configuration.
\RequirePackage{titlesec}
\RequirePackage{apptools}
%% Change the chapter style.
\titleformat{\chapter}[hang]{\LARGE\bfseries}{\IfAppendix{\appendixname{} }{}\thechapter}{1em}{}% NEW
\titlespacing*{\chapter}{0pt}{0pt}{12pt}

%% Remove the spacing between the title and the border.
\makeatletter
\def\ttl@mkchap@i#1#2#3#4#5#6#7{%
    \ttl@assign\@tempskipa#3\relax\beforetitleunit
    \vspace{\@tempskipa}%
    \global\@afterindenttrue
    \ifcase#5 \global\@afterindentfalse\fi
    \ttl@assign\@tempskipb#4\relax\aftertitleunit
    \ttl@topmode{\@tempskipb}{%
        \ttl@select{#6}{#1}{#2}{#7}}%
    \ttl@finmarks  % Outside the box!
    \@ifundefined{ttlp@#6}{}{\ttlp@write{#6}}}
\makeatother

% Table of contents configuration.
\RequirePackage{tocloft}
%% List of figures configuration.
\renewcommand{\cftfigfont}{Figure }
\renewcommand{\cftfigaftersnum}{}
\renewcommand{\cftfigindent}{0pt}
%% List of tables configuration.
\renewcommand{\cfttabaftersnum}{}
\renewcommand{\cfttabfont}{Table }
\renewcommand{\cfttabindent}{0pt}

% Caption configuration.
\RequirePackage[justification=centering]{caption}
\captionsetup{labelsep=space}

% Algorithm environment.
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}

% Add color support.
\RequirePackage{xcolor}

% Add listings support.
\RequirePackage{listings}
%% Define custom listing style.
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2
}
\lstset{style=mystyle}

% Add Corollary, theorems, and lemmas support
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{corollary}{Corollary}[theorem]
\newtheorem{lemma}[theorem]{Lemma}

% Glossary and acronym mangement.
\RequirePackage[automake,acronym]{glossaries}
%% Remove the bold font when listing abbreviations and glossary terms.
\renewcommand*{\glsnamefont}[1]{\textmd{#1}}
%% Command required to generate glossaries.
\makeglossaries{}

% Header & Footer configuration.
\RequirePackage{fancyhdr}

%% Override the chapter and section marks of the default 'fancy' style.
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\renewcommand{\sectionmark}[1]{\markright{#1}}

%% Create a style that only shows page numbers.
\fancypagestyle{pageNumbersOnly}
{
    \fancyhf{}
    \rhead{}
    \lhead{}
    \cfoot{\fancyplain{}{\thepage}}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
}

%% Update the Main style.
\fancypagestyle{MainMatter}
{
    \fancyhf{}
    \lhead{}
    \rhead{\fancyplain{}{\thechapter{} \leftmark}}
    \cfoot{\fancyplain{}{\thepage}}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
}

%% Set the header to a fixed height of 15.8pt.
\setlength{\headheight}{15.8pt}

% Custom commands.
\makeatletter
\newcommand\supervisor[1]{\newcommand\@supervisor{#1}}
\newcommand\cosupervisor[1]{\newcommand\@cosupervisor{#1}}
\newcommand\titledate[1]{\newcommand\@titledate{#1}}
\newcommand\degreename[1]{\newcommand\@degreename{#1}}
\makeatother
