import {
  flowRendererV2,
  flowStyles
} from "./chunk-HZBYD3V5.js";
import {
  flowDb,
  parser$1
} from "./chunk-SO5ROR5X.js";
import "./chunk-R76AYJDD.js";
import "./chunk-4L4CY6G2.js";
import "./chunk-MGFSAB4W.js";
import "./chunk-JDGIDPOV.js";
import "./chunk-VSKK27YF.js";
import {
  require_dayjs_min,
  require_dist,
  setConfig
} from "./chunk-C32FXNKI.js";
import {
  __toESM
} from "./chunk-PR4QN5HX.js";

// node_modules/mermaid/dist/flowDiagram-v2-96b9c2cf.js
var import_dayjs = __toESM(require_dayjs_min(), 1);
var import_sanitize_url = __toESM(require_dist(), 1);
var diagram = {
  parser: parser$1,
  db: flowDb,
  renderer: flowRendererV2,
  styles: flowStyles,
  init: (cnf) => {
    if (!cnf.flowchart) {
      cnf.flowchart = {};
    }
    cnf.flowchart.arrowMarkerAbsolute = cnf.arrowMarkerAbsolute;
    setConfig({ flowchart: { arrowMarkerAbsolute: cnf.arrowMarkerAbsolute } });
    flowRendererV2.setConf(cnf.flowchart);
    flowDb.clear();
    flowDb.setGen("gen-2");
  }
};
export {
  diagram
};
//# sourceMappingURL=flowDiagram-v2-96b9c2cf-HNZUHTXB.js.map
