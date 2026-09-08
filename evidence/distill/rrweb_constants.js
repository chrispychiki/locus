// Auto-generated from @rrweb/types@2.1.1 by distill/extract_rrweb_constants.js
// DO NOT EDIT — regenerate with: bun distill/extract_rrweb_constants.js

export const EventType = {
  DomContentLoaded: 0,
  Load: 1,
  FullSnapshot: 2,
  IncrementalSnapshot: 3,
  Meta: 4,
  Custom: 5,
  Plugin: 6,
  Asset: 7,
  PageLoad: 69,
  PageVisible: 70,
  PageHidden: 71,
};

export const EventTypeNames = {
  0: "DomContentLoaded",
  1: "Load",
  2: "FullSnapshot",
  3: "IncrementalSnapshot",
  4: "Meta",
  5: "Custom",
  6: "Plugin",
  7: "Asset",
  69: "PageLoad",
  70: "PageVisible",
  71: "PageHidden",
};

export const IncrementalSource = {
  Mutation: 0,
  MouseMove: 1,
  MouseInteraction: 2,
  Scroll: 3,
  ViewportResize: 4,
  Input: 5,
  TouchMove: 6,
  MediaInteraction: 7,
  StyleSheetRule: 8,
  CanvasMutation: 9,
  Font: 10,
  Log: 11,
  Drag: 12,
  StyleDeclaration: 13,
  Selection: 14,
  AdoptedStyleSheet: 15,
  CustomElement: 16,
};

export const IncrementalSourceNames = {
  0: "Mutation",
  1: "MouseMove",
  2: "MouseInteraction",
  3: "Scroll",
  4: "ViewportResize",
  5: "Input",
  6: "TouchMove",
  7: "MediaInteraction",
  8: "StyleSheetRule",
  9: "CanvasMutation",
  10: "Font",
  11: "Log",
  12: "Drag",
  13: "StyleDeclaration",
  14: "Selection",
  15: "AdoptedStyleSheet",
  16: "CustomElement",
};

export const MediaInteractions = {
  Play: 0,
  Pause: 1,
  Seeked: 2,
  VolumeChange: 3,
  RateChange: 4,
};

export const MediaInteractionsNames = {
  0: "Play",
  1: "Pause",
  2: "Seeked",
  3: "VolumeChange",
  4: "RateChange",
};

export const MouseInteractions = {
  MouseUp: 0,
  MouseDown: 1,
  Click: 2,
  ContextMenu: 3,
  DblClick: 4,
  Focus: 5,
  Blur: 6,
  TouchStart: 7,
  TouchMove_Departed: 8,
  TouchEnd: 9,
  TouchCancel: 10,
};

export const MouseInteractionsNames = {
  0: "MouseUp",
  1: "MouseDown",
  2: "Click",
  3: "ContextMenu",
  4: "DblClick",
  5: "Focus",
  6: "Blur",
  7: "TouchStart",
  8: "TouchMove_Departed",
  9: "TouchEnd",
  10: "TouchCancel",
};

export const PointerTypes = {
  Mouse: 0,
  Pen: 1,
  Touch: 2,
};

export const PointerTypesNames = {
  0: "Mouse",
  1: "Pen",
  2: "Touch",
};

export const NodeType = {
  Document: 0,
  DocumentType: 1,
  Element: 2,
  Text: 3,
  CDATA: 4,
  Comment: 5,
};

export const NodeTypeNames = {
  0: "Document",
  1: "DocumentType",
  2: "Element",
  3: "Text",
  4: "CDATA",
  5: "Comment",
};

export const SVGTagMap = {
  altglyph: "altGlyph",
  altglyphdef: "altGlyphDef",
  altglyphitem: "altGlyphItem",
  animatecolor: "animateColor",
  animatemotion: "animateMotion",
  animatetransform: "animateTransform",
  clippath: "clipPath",
  feblend: "feBlend",
  fecolormatrix: "feColorMatrix",
  fecomponenttransfer: "feComponentTransfer",
  fecomposite: "feComposite",
  feconvolvematrix: "feConvolveMatrix",
  fediffuselighting: "feDiffuseLighting",
  fedisplacementmap: "feDisplacementMap",
  fedistantlight: "feDistantLight",
  fedropshadow: "feDropShadow",
  feflood: "feFlood",
  fefunca: "feFuncA",
  fefuncb: "feFuncB",
  fefuncg: "feFuncG",
  fefuncr: "feFuncR",
  fegaussianblur: "feGaussianBlur",
  feimage: "feImage",
  femerge: "feMerge",
  femergenode: "feMergeNode",
  femorphology: "feMorphology",
  feoffset: "feOffset",
  fepointlight: "fePointLight",
  fespecularlighting: "feSpecularLighting",
  fespotlight: "feSpotLight",
  fetile: "feTile",
  feturbulence: "feTurbulence",
  foreignobject: "foreignObject",
  glyphref: "glyphRef",
  lineargradient: "linearGradient",
  radialgradient: "radialGradient",
};
