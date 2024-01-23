'use strict';

var obsidian = require('obsidian');

/*! *****************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
/* global Reflect, Promise */

var extendStatics = function(d, b) {
    extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
    return extendStatics(d, b);
};

function __extends(d, b) {
    extendStatics(d, b);
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}

function __awaiter(thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
}

function __generator(thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
}

function __spreadArrays() {
    for (var s = 0, i = 0, il = arguments.length; i < il; i++) s += arguments[i].length;
    for (var r = Array(s), k = 0, i = 0; i < il; i++)
        for (var a = arguments[i], j = 0, jl = a.length; j < jl; j++, k++)
            r[k] = a[j];
    return r;
}

var LatexEnvironmentsSettings = /** @class */ (function () {
    function LatexEnvironmentsSettings() {
        this.defaultEnvironment = 'multline';
        this.customEnvironments = [];
    }
    return LatexEnvironmentsSettings;
}());
function ensureSettings(loaded) {
    var _a, _b;
    var settings = new LatexEnvironmentsSettings();
    settings.defaultEnvironment = (_a = loaded.defaultEnvironment) !== null && _a !== void 0 ? _a : settings.defaultEnvironment;
    settings.customEnvironments = (_b = loaded.customEnvironments) !== null && _b !== void 0 ? _b : settings.customEnvironments;
    return settings;
}

var BEGIN_LENGTH = 8;
var END_LENGTH = 6;
var Environment = /** @class */ (function () {
    function Environment(doc, _name, _start, _end) {
        this.doc = doc;
        this._name = _name;
        this._start = _start;
        this._end = _end;
    }
    Object.defineProperty(Environment.prototype, "name", {
        get: function () {
            return this._name;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Environment.prototype, "start", {
        get: function () {
            return this._start;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Environment.prototype, "end", {
        get: function () {
            return this._end;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Environment.prototype, "beginString", {
        get: function () {
            return "\\begin{" + this._name + "}";
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Environment.prototype, "endString", {
        get: function () {
            return "\\end{" + this._name + "}";
        },
        enumerable: false,
        configurable: true
    });
    Environment.prototype.replace = function (envName) {
        this._name = envName;
        this.doc.replaceRange(this.beginString, this.start.from, this.start.to);
        this.doc.replaceRange(this.endString, this.end.from, this.end.to);
        this._start.to = {
            line: this.start.from.line,
            ch: this.start.from.ch + this.beginString.length,
        };
        this._end.to = {
            line: this.end.from.line,
            ch: this.end.from.ch + this.endString.length,
        };
        return this;
    };
    Environment.prototype.print = function (contents) {
        if (contents === void 0) { contents = '\n\n'; }
        return "" + this.beginString + contents + this.endString;
    };
    Environment.newRange = function (cursor, envName, lineOffset, chOffset) {
        return {
            from: {
                line: cursor.line + lineOffset,
                ch: cursor.ch,
            },
            to: {
                line: cursor.line + lineOffset,
                ch: cursor.ch + chOffset + envName.length,
            },
        };
    };
    Environment.create = function (envName, doc, cursor) {
        var newLine = nextLine(cursor, true);
        var newEnvironment = new Environment(doc, envName, this.newRange(newLine, envName, 0, BEGIN_LENGTH), this.newRange(newLine, envName, 2, END_LENGTH));
        var line = doc.getLine(cursor.line);
        var frontPad = getPad(line.substr(0, cursor.ch));
        var rearPad = getPad(line.substr(cursor.ch));
        doc.replaceRange(frontPad + newEnvironment.print() + rearPad, cursor);
        doc.setCursor(nextLine(newLine, false, frontPad.length));
        return newEnvironment;
    };
    Environment.wrap = function (envName, doc, from, to, outerPad) {
        if (outerPad === void 0) { outerPad = '\n'; }
        var newEnvironment = new Environment(doc, envName, this.newRange(from, envName, 0, BEGIN_LENGTH), this.newRange(nextLine(to, true, 2), envName, 2, END_LENGTH));
        doc.replaceRange(outerPad + newEnvironment.endString, to);
        doc.replaceRange(newEnvironment.beginString + outerPad, from);
        if (doc.somethingSelected()) {
            doc.setCursor(nextLine(from, true));
        }
        else {
            var lineOffset = newEnvironment.start.from.line - from.line;
            doc.setCursor(nextLine(doc.getCursor(), false, lineOffset));
        }
        return newEnvironment;
    };
    Environment.prototype.unwrap = function () {
        this.doc.replaceRange('', this.start.from, this.start.to);
        this.doc.replaceRange('', this.end.from, this.end.to);
    };
    return Environment;
}());
function nextLine(cursor, cr, offset) {
    if (cr === void 0) { cr = false; }
    if (offset === void 0) { offset = 1; }
    return { line: cursor.line + offset, ch: cr ? 0 : cursor.ch };
}
function getPad(text) {
    if (text.match(/^[ \t]*$/) !== null) {
        return '';
    }
    return '\n';
}

var MathBlock = /** @class */ (function () {
    function MathBlock(doc, cursor) {
        var searchCursor = doc.getSearchCursor('$$', cursor);
        this.startPosition =
            searchCursor.findPrevious() !== false
                ? searchCursor.to()
                : { line: doc.firstLine(), ch: 0 };
        this.endPosition =
            searchCursor.findNext() !== false
                ? searchCursor.from()
                : { line: doc.lastLine(), ch: doc.getLine(doc.lastLine()).length - 1 };
        this.doc = doc;
    }
    MathBlock.prototype.getEnclosingEnvironment = function (cursor) {
        var beginEnds = new BeginEnds(this.doc, this.startPosition, this.endPosition);
        var environments = Array.from(beginEnds);
        if (beginEnds.isOpen) {
            throw new Error('unclosed environments in block');
        }
        var start = environments
            .filter(function (env) {
            var from = env.pos.from;
            return (env.type === 'begin' &&
                (from.line < cursor.line ||
                    (from.line === cursor.line && from.ch <= cursor.ch)));
        })
            .pop();
        if (start === undefined) {
            return undefined;
        }
        var startTo = start.pos.to;
        var after = environments.filter(function (env) {
            var from = env.pos.from;
            return (from.line > startTo.line ||
                (from.line === startTo.line && from.ch > startTo.ch));
        });
        var open = 1;
        var end;
        for (var _i = 0, after_1 = after; _i < after_1.length; _i++) {
            var env = after_1[_i];
            if (env.type === 'begin') {
                open++;
            }
            else {
                open--;
                if (open === 0) {
                    end = env;
                    break;
                }
            }
        }
        if (end === undefined) {
            throw new Error('current environment is never closed');
        }
        var endTo = end.pos.to;
        if (endTo.line < cursor.line ||
            (endTo.line === cursor.line && endTo.ch < cursor.ch)) {
            return undefined;
        }
        return new Environment(this.doc, start.name, start.pos, end.pos);
    };
    MathBlock.isMathMode = function (cursor, editor) {
        var token = editor.getTokenAt(cursor);
        var state = token.state;
        return state.hmdInnerStyle === 'math';
    };
    return MathBlock;
}());
var BeginEnds = /** @class */ (function () {
    function BeginEnds(doc, start, end) {
        this.doc = doc;
        this.start = start;
        this.end = end;
        this.openEnvs = [];
        this.search = this.getEnvCursor(this.start);
    }
    BeginEnds.prototype.reset = function () {
        this.search = this.getEnvCursor(this.start);
    };
    BeginEnds.prototype.getEnvCursor = function (start) {
        return this.doc.getSearchCursor(/\\(begin|end){\s*([^}]+)\s*}/m, start);
    };
    Object.defineProperty(BeginEnds.prototype, "isOpen", {
        get: function () {
            return this.openEnvs.length > 0;
        },
        enumerable: false,
        configurable: true
    });
    BeginEnds.prototype[Symbol.iterator] = function () {
        this.reset();
        return this;
    };
    BeginEnds.prototype.next = function () {
        var match = this.search.findNext();
        var to = this.search.to();
        if (match === true ||
            match === false ||
            to.line > this.end.line ||
            (to.line === this.end.line && to.ch > this.end.ch)) {
            return { done: true, value: null };
        }
        switch (match[1]) {
            case 'begin': {
                var current = {
                    name: match[2],
                    type: 'begin',
                    pos: {
                        from: this.search.from(),
                        to: this.search.to(),
                    },
                };
                this.openEnvs.push(current);
                return {
                    done: false,
                    value: current,
                };
            }
            case 'end': {
                var current = this.openEnvs.pop();
                if (current === undefined) {
                    throw new Error('closing environment which was never opened');
                }
                if (current.name !== match[2]) {
                    throw new Error('environment not closed properly');
                }
                return {
                    done: false,
                    value: {
                        name: match[2],
                        type: 'end',
                        pos: {
                            from: this.search.from(),
                            to: this.search.to(),
                        },
                    },
                };
            }
        }
        throw new Error("regex returned unexpected result " + match[1]);
    };
    return BeginEnds;
}());

var DISPLAY_EQUATIONS = [
    'equation',
    'equation*',
    'gather',
    'gather*',
    'multline',
    'multline*',
    'split',
    'align',
    'align*',
    'flalign',
    'flalign*',
    'alignat',
    'alignat*',
];
var MATRICES = [
    'matrix',
    'pmatrix',
    'bmatrix',
    'Bmatrix',
    'vmatrix',
    'Vmatrix',
    'smallmatrix',
];
var SUB_ENVIRONMENTS = ['multlined', 'gathered', 'aligned', 'cases'];
var DEFAULT_ENVIRONMENTS = __spreadArrays(DISPLAY_EQUATIONS, MATRICES, SUB_ENVIRONMENTS);

var EnvModal = /** @class */ (function (_super) {
    __extends(EnvModal, _super);
    function EnvModal(app, settings, name, callback) {
        var _this = _super.call(this, app) || this;
        _this.settings = settings;
        _this.name = name;
        _this.callback = callback;
        _this.matched = false;
        _this.setInstructions([
            { command: '↑↓', purpose: 'to navigate' },
            { command: '↵', purpose: 'to select' },
            { command: 'esc', purpose: 'to dismiss' },
        ]);
        _this.setPlaceholder('environment name');
        return _this;
    }
    EnvModal.prototype.getItems = function () {
        return Array.from(new Set([this.settings.defaultEnvironment].concat(this.settings.customEnvironments, DEFAULT_ENVIRONMENTS)));
    };
    EnvModal.prototype.getItemText = function (item) {
        this.matched = true;
        return item;
    };
    EnvModal.prototype.onNoSuggestion = function () {
        this.matched = false;
    };
    EnvModal.prototype.onChooseItem = function (item, _evt) {
        if (this.matched) {
            this.callback(item);
        }
        else {
            this.callback(this.inputEl.value);
        }
    };
    EnvModal.callback = function (app, settings, defaultName, call) {
        new EnvModal(app, settings, defaultName, call).open();
    };
    return EnvModal;
}(obsidian.FuzzySuggestModal));

var LatexEnvironmentsSettingTab = /** @class */ (function (_super) {
    __extends(LatexEnvironmentsSettingTab, _super);
    function LatexEnvironmentsSettingTab(app, plugin) {
        var _this = _super.call(this, app, plugin) || this;
        _this.plugin = plugin;
        return _this;
    }
    LatexEnvironmentsSettingTab.prototype.display = function () {
        var _this = this;
        var containerEl = this.containerEl;
        containerEl.empty();
        containerEl.createEl('h2', { text: 'Settings for latex environments' });
        new obsidian.Setting(containerEl)
            .setName('Default environment')
            .setDesc('The default environment to insert')
            .addText(function (text) {
            return text
                .setPlaceholder('environment')
                .setValue(_this.plugin.settings.defaultEnvironment)
                .onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.plugin.settings.defaultEnvironment = value;
                            return [4 /*yield*/, this.plugin.saveData(this.plugin.settings)];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
        new obsidian.Setting(containerEl)
            .setName('Extra environments')
            .setDesc('Environment names to be suggested for completion (one per line)')
            .addTextArea(function (area) {
            area
                .setValue(_this.plugin.settings.customEnvironments.join('\n'))
                .onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.plugin.settings.customEnvironments = value
                                .split('\n')
                                .map(function (x) { return x.trim(); })
                                .filter(function (x) { return x.length > 0; });
                            return [4 /*yield*/, this.plugin.saveData(this.plugin.settings)];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
    };
    return LatexEnvironmentsSettingTab;
}(obsidian.PluginSettingTab));

var Action = /** @class */ (function () {
    function Action(doc) {
        this.doc = doc;
    }
    Action.prototype.suggestName = function () {
        return undefined;
    };
    Object.defineProperty(Action.prototype, "needsName", {
        get: function () {
            return true;
        },
        enumerable: false,
        configurable: true
    });
    return Action;
}());

var WrapAction = /** @class */ (function (_super) {
    __extends(WrapAction, _super);
    function WrapAction(doc, from, to, addWhitespace) {
        if (addWhitespace === void 0) { addWhitespace = true; }
        var _this = _super.call(this, doc) || this;
        _this.from = from;
        _this.to = to;
        _this.addWhitespace = addWhitespace;
        return _this;
    }
    WrapAction.prototype.prepare = function () {
        return this;
    };
    WrapAction.prototype.execute = function (envName) {
        Environment.wrap(envName, this.doc, this.from, this.to, this.addWhitespace ? '\n' : '');
    };
    return WrapAction;
}(Action));

var InsertAction = /** @class */ (function (_super) {
    __extends(InsertAction, _super);
    function InsertAction() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    InsertAction.prototype.prepare = function () {
        if (this.doc.somethingSelected()) {
            return new WrapAction(this.doc, this.doc.getCursor('from'), this.doc.getCursor('to')).prepare();
        }
        return this;
    };
    InsertAction.prototype.execute = function (envName) {
        Environment.create(envName, this.doc, this.doc.getCursor());
    };
    return InsertAction;
}(Action));

var ChangeAction = /** @class */ (function (_super) {
    __extends(ChangeAction, _super);
    function ChangeAction() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    ChangeAction.prototype.suggestName = function () {
        return this.name;
    };
    ChangeAction.prototype.prepare = function () {
        var cursor = this.doc.getCursor();
        var block = new MathBlock(this.doc, cursor);
        this.current = block.getEnclosingEnvironment(cursor);
        if (this.current === undefined) {
            return new WrapAction(this.doc, block.startPosition, block.endPosition, block.startPosition.line === block.endPosition.line);
        }
        this.name = this.current.name;
        return this;
    };
    ChangeAction.prototype.execute = function (envName) {
        if (this.current !== undefined)
            this.current.replace(envName);
    };
    return ChangeAction;
}(Action));

var DeleteAction = /** @class */ (function (_super) {
    __extends(DeleteAction, _super);
    function DeleteAction() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Object.defineProperty(DeleteAction.prototype, "needsName", {
        get: function () {
            return false;
        },
        enumerable: false,
        configurable: true
    });
    DeleteAction.prototype.prepare = function () {
        var cursor = this.doc.getCursor();
        var block = new MathBlock(this.doc, cursor);
        this.current = block.getEnclosingEnvironment(cursor);
        return this;
    };
    DeleteAction.prototype.execute = function (_envName) {
        if (this.current !== undefined)
            this.current.unwrap();
    };
    return DeleteAction;
}(Action));

var LatexEnvironments = /** @class */ (function (_super) {
    __extends(LatexEnvironments, _super);
    function LatexEnvironments() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.settings = new LatexEnvironmentsSettings();
        return _this;
    }
    LatexEnvironments.prototype.onload = function () {
        return __awaiter(this, void 0, void 0, function () {
            var settings;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.loadData()];
                    case 1:
                        settings = _a.sent();
                        if (settings !== null) {
                            this.settings = ensureSettings(settings);
                        }
                        this.addCommand({
                            id: 'insert-latex-env',
                            name: 'Insert LaTeX environment',
                            checkCallback: this.mathModeCallback(InsertAction),
                        });
                        this.addCommand({
                            id: 'change-latex-env',
                            name: 'Change LaTeX environment',
                            checkCallback: this.mathModeCallback(ChangeAction),
                        });
                        this.addCommand({
                            id: 'delete-latex-env',
                            name: 'Delete LaTeX environment',
                            checkCallback: this.mathModeCallback(DeleteAction),
                        });
                        this.addSettingTab(new LatexEnvironmentsSettingTab(this.app, this));
                        return [2 /*return*/];
                }
            });
        });
    };
    LatexEnvironments.prototype.mathModeCallback = function (ActionType) {
        var _this = this;
        return function (checking) {
            var leaf = _this.app.workspace.activeLeaf;
            if (leaf.view instanceof obsidian.MarkdownView) {
                var editor = leaf.view.sourceMode.cmEditor;
                var cursor = editor.getCursor();
                if (!MathBlock.isMathMode(cursor, editor)) {
                    return false;
                }
                if (!checking) {
                    try {
                        var action = new ActionType(editor.getDoc()).prepare();
                        _this.withPromptName(editor, action);
                    }
                    catch (e) {
                        /* eslint-disable-next-line no-new */
                        new obsidian.Notice(e.message);
                    }
                }
                return true;
            }
            return false;
        };
    };
    LatexEnvironments.prototype.withPromptName = function (editor, action) {
        var call = function (envName) {
            editor.operation(function () { return action.execute(envName); });
            editor.focus();
        };
        if (action.needsName) {
            var suggested = action.suggestName();
            EnvModal.callback(this.app, this.settings, suggested !== undefined ? suggested : this.settings.defaultEnvironment, call);
        }
        else {
            call('*');
        }
    };
    return LatexEnvironments;
}(obsidian.Plugin));

module.exports = LatexEnvironments;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWFpbi5qcyIsInNvdXJjZXMiOlsiLi4vbm9kZV9tb2R1bGVzLy5wbnBtL0Byb2xsdXAvcGx1Z2luLXR5cGVzY3JpcHRANi4xLjBfNjZjMDIyMWIzY2RlOTg1YzI1MTIwOTFhMmU4NDZhMDQvbm9kZV9tb2R1bGVzL3RzbGliL3RzbGliLmVzNi5qcyIsIi4uL3NyYy9zZXR0aW5ncy50cyIsIi4uL3NyYy9lbnZpcm9ubWVudC50cyIsIi4uL3NyYy9tYXRoYmxvY2sudHMiLCIuLi9zcmMvZW52aXJvbm1lbnROYW1lcy50cyIsIi4uL3NyYy9lbnZtb2RhbC50cyIsIi4uL3NyYy9sYXRleEVudmlyb25tZW50c1NldHRpbmdzVGFiLnRzIiwiLi4vc3JjL2FjdGlvbnMvYWN0aW9uLnRzIiwiLi4vc3JjL2FjdGlvbnMvd3JhcEFjdGlvbi50cyIsIi4uL3NyYy9hY3Rpb25zL2luc2VydEFjdGlvbi50cyIsIi4uL3NyYy9hY3Rpb25zL2NoYW5nZUFjdGlvbi50cyIsIi4uL3NyYy9hY3Rpb25zL2RlbGV0ZUFjdGlvbi50cyIsIi4uL3NyYy9tYWluLnRzIl0sInNvdXJjZXNDb250ZW50IjpbIi8qISAqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxyXG5Db3B5cmlnaHQgKGMpIE1pY3Jvc29mdCBDb3Jwb3JhdGlvbi5cclxuXHJcblBlcm1pc3Npb24gdG8gdXNlLCBjb3B5LCBtb2RpZnksIGFuZC9vciBkaXN0cmlidXRlIHRoaXMgc29mdHdhcmUgZm9yIGFueVxyXG5wdXJwb3NlIHdpdGggb3Igd2l0aG91dCBmZWUgaXMgaGVyZWJ5IGdyYW50ZWQuXHJcblxyXG5USEUgU09GVFdBUkUgSVMgUFJPVklERUQgXCJBUyBJU1wiIEFORCBUSEUgQVVUSE9SIERJU0NMQUlNUyBBTEwgV0FSUkFOVElFUyBXSVRIXHJcblJFR0FSRCBUTyBUSElTIFNPRlRXQVJFIElOQ0xVRElORyBBTEwgSU1QTElFRCBXQVJSQU5USUVTIE9GIE1FUkNIQU5UQUJJTElUWVxyXG5BTkQgRklUTkVTUy4gSU4gTk8gRVZFTlQgU0hBTEwgVEhFIEFVVEhPUiBCRSBMSUFCTEUgRk9SIEFOWSBTUEVDSUFMLCBESVJFQ1QsXHJcbklORElSRUNULCBPUiBDT05TRVFVRU5USUFMIERBTUFHRVMgT1IgQU5ZIERBTUFHRVMgV0hBVFNPRVZFUiBSRVNVTFRJTkcgRlJPTVxyXG5MT1NTIE9GIFVTRSwgREFUQSBPUiBQUk9GSVRTLCBXSEVUSEVSIElOIEFOIEFDVElPTiBPRiBDT05UUkFDVCwgTkVHTElHRU5DRSBPUlxyXG5PVEhFUiBUT1JUSU9VUyBBQ1RJT04sIEFSSVNJTkcgT1VUIE9GIE9SIElOIENPTk5FQ1RJT04gV0lUSCBUSEUgVVNFIE9SXHJcblBFUkZPUk1BTkNFIE9GIFRISVMgU09GVFdBUkUuXHJcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqICovXHJcbi8qIGdsb2JhbCBSZWZsZWN0LCBQcm9taXNlICovXHJcblxyXG52YXIgZXh0ZW5kU3RhdGljcyA9IGZ1bmN0aW9uKGQsIGIpIHtcclxuICAgIGV4dGVuZFN0YXRpY3MgPSBPYmplY3Quc2V0UHJvdG90eXBlT2YgfHxcclxuICAgICAgICAoeyBfX3Byb3RvX186IFtdIH0gaW5zdGFuY2VvZiBBcnJheSAmJiBmdW5jdGlvbiAoZCwgYikgeyBkLl9fcHJvdG9fXyA9IGI7IH0pIHx8XHJcbiAgICAgICAgZnVuY3Rpb24gKGQsIGIpIHsgZm9yICh2YXIgcCBpbiBiKSBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKGIsIHApKSBkW3BdID0gYltwXTsgfTtcclxuICAgIHJldHVybiBleHRlbmRTdGF0aWNzKGQsIGIpO1xyXG59O1xyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fZXh0ZW5kcyhkLCBiKSB7XHJcbiAgICBleHRlbmRTdGF0aWNzKGQsIGIpO1xyXG4gICAgZnVuY3Rpb24gX18oKSB7IHRoaXMuY29uc3RydWN0b3IgPSBkOyB9XHJcbiAgICBkLnByb3RvdHlwZSA9IGIgPT09IG51bGwgPyBPYmplY3QuY3JlYXRlKGIpIDogKF9fLnByb3RvdHlwZSA9IGIucHJvdG90eXBlLCBuZXcgX18oKSk7XHJcbn1cclxuXHJcbmV4cG9ydCB2YXIgX19hc3NpZ24gPSBmdW5jdGlvbigpIHtcclxuICAgIF9fYXNzaWduID0gT2JqZWN0LmFzc2lnbiB8fCBmdW5jdGlvbiBfX2Fzc2lnbih0KSB7XHJcbiAgICAgICAgZm9yICh2YXIgcywgaSA9IDEsIG4gPSBhcmd1bWVudHMubGVuZ3RoOyBpIDwgbjsgaSsrKSB7XHJcbiAgICAgICAgICAgIHMgPSBhcmd1bWVudHNbaV07XHJcbiAgICAgICAgICAgIGZvciAodmFyIHAgaW4gcykgaWYgKE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChzLCBwKSkgdFtwXSA9IHNbcF07XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiB0O1xyXG4gICAgfVxyXG4gICAgcmV0dXJuIF9fYXNzaWduLmFwcGx5KHRoaXMsIGFyZ3VtZW50cyk7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3Jlc3QocywgZSkge1xyXG4gICAgdmFyIHQgPSB7fTtcclxuICAgIGZvciAodmFyIHAgaW4gcykgaWYgKE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChzLCBwKSAmJiBlLmluZGV4T2YocCkgPCAwKVxyXG4gICAgICAgIHRbcF0gPSBzW3BdO1xyXG4gICAgaWYgKHMgIT0gbnVsbCAmJiB0eXBlb2YgT2JqZWN0LmdldE93blByb3BlcnR5U3ltYm9scyA9PT0gXCJmdW5jdGlvblwiKVxyXG4gICAgICAgIGZvciAodmFyIGkgPSAwLCBwID0gT2JqZWN0LmdldE93blByb3BlcnR5U3ltYm9scyhzKTsgaSA8IHAubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgaWYgKGUuaW5kZXhPZihwW2ldKSA8IDAgJiYgT2JqZWN0LnByb3RvdHlwZS5wcm9wZXJ0eUlzRW51bWVyYWJsZS5jYWxsKHMsIHBbaV0pKVxyXG4gICAgICAgICAgICAgICAgdFtwW2ldXSA9IHNbcFtpXV07XHJcbiAgICAgICAgfVxyXG4gICAgcmV0dXJuIHQ7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2RlY29yYXRlKGRlY29yYXRvcnMsIHRhcmdldCwga2V5LCBkZXNjKSB7XHJcbiAgICB2YXIgYyA9IGFyZ3VtZW50cy5sZW5ndGgsIHIgPSBjIDwgMyA/IHRhcmdldCA6IGRlc2MgPT09IG51bGwgPyBkZXNjID0gT2JqZWN0LmdldE93blByb3BlcnR5RGVzY3JpcHRvcih0YXJnZXQsIGtleSkgOiBkZXNjLCBkO1xyXG4gICAgaWYgKHR5cGVvZiBSZWZsZWN0ID09PSBcIm9iamVjdFwiICYmIHR5cGVvZiBSZWZsZWN0LmRlY29yYXRlID09PSBcImZ1bmN0aW9uXCIpIHIgPSBSZWZsZWN0LmRlY29yYXRlKGRlY29yYXRvcnMsIHRhcmdldCwga2V5LCBkZXNjKTtcclxuICAgIGVsc2UgZm9yICh2YXIgaSA9IGRlY29yYXRvcnMubGVuZ3RoIC0gMTsgaSA+PSAwOyBpLS0pIGlmIChkID0gZGVjb3JhdG9yc1tpXSkgciA9IChjIDwgMyA/IGQocikgOiBjID4gMyA/IGQodGFyZ2V0LCBrZXksIHIpIDogZCh0YXJnZXQsIGtleSkpIHx8IHI7XHJcbiAgICByZXR1cm4gYyA+IDMgJiYgciAmJiBPYmplY3QuZGVmaW5lUHJvcGVydHkodGFyZ2V0LCBrZXksIHIpLCByO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19wYXJhbShwYXJhbUluZGV4LCBkZWNvcmF0b3IpIHtcclxuICAgIHJldHVybiBmdW5jdGlvbiAodGFyZ2V0LCBrZXkpIHsgZGVjb3JhdG9yKHRhcmdldCwga2V5LCBwYXJhbUluZGV4KTsgfVxyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19tZXRhZGF0YShtZXRhZGF0YUtleSwgbWV0YWRhdGFWYWx1ZSkge1xyXG4gICAgaWYgKHR5cGVvZiBSZWZsZWN0ID09PSBcIm9iamVjdFwiICYmIHR5cGVvZiBSZWZsZWN0Lm1ldGFkYXRhID09PSBcImZ1bmN0aW9uXCIpIHJldHVybiBSZWZsZWN0Lm1ldGFkYXRhKG1ldGFkYXRhS2V5LCBtZXRhZGF0YVZhbHVlKTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fYXdhaXRlcih0aGlzQXJnLCBfYXJndW1lbnRzLCBQLCBnZW5lcmF0b3IpIHtcclxuICAgIGZ1bmN0aW9uIGFkb3B0KHZhbHVlKSB7IHJldHVybiB2YWx1ZSBpbnN0YW5jZW9mIFAgPyB2YWx1ZSA6IG5ldyBQKGZ1bmN0aW9uIChyZXNvbHZlKSB7IHJlc29sdmUodmFsdWUpOyB9KTsgfVxyXG4gICAgcmV0dXJuIG5ldyAoUCB8fCAoUCA9IFByb21pc2UpKShmdW5jdGlvbiAocmVzb2x2ZSwgcmVqZWN0KSB7XHJcbiAgICAgICAgZnVuY3Rpb24gZnVsZmlsbGVkKHZhbHVlKSB7IHRyeSB7IHN0ZXAoZ2VuZXJhdG9yLm5leHQodmFsdWUpKTsgfSBjYXRjaCAoZSkgeyByZWplY3QoZSk7IH0gfVxyXG4gICAgICAgIGZ1bmN0aW9uIHJlamVjdGVkKHZhbHVlKSB7IHRyeSB7IHN0ZXAoZ2VuZXJhdG9yW1widGhyb3dcIl0odmFsdWUpKTsgfSBjYXRjaCAoZSkgeyByZWplY3QoZSk7IH0gfVxyXG4gICAgICAgIGZ1bmN0aW9uIHN0ZXAocmVzdWx0KSB7IHJlc3VsdC5kb25lID8gcmVzb2x2ZShyZXN1bHQudmFsdWUpIDogYWRvcHQocmVzdWx0LnZhbHVlKS50aGVuKGZ1bGZpbGxlZCwgcmVqZWN0ZWQpOyB9XHJcbiAgICAgICAgc3RlcCgoZ2VuZXJhdG9yID0gZ2VuZXJhdG9yLmFwcGx5KHRoaXNBcmcsIF9hcmd1bWVudHMgfHwgW10pKS5uZXh0KCkpO1xyXG4gICAgfSk7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2dlbmVyYXRvcih0aGlzQXJnLCBib2R5KSB7XHJcbiAgICB2YXIgXyA9IHsgbGFiZWw6IDAsIHNlbnQ6IGZ1bmN0aW9uKCkgeyBpZiAodFswXSAmIDEpIHRocm93IHRbMV07IHJldHVybiB0WzFdOyB9LCB0cnlzOiBbXSwgb3BzOiBbXSB9LCBmLCB5LCB0LCBnO1xyXG4gICAgcmV0dXJuIGcgPSB7IG5leHQ6IHZlcmIoMCksIFwidGhyb3dcIjogdmVyYigxKSwgXCJyZXR1cm5cIjogdmVyYigyKSB9LCB0eXBlb2YgU3ltYm9sID09PSBcImZ1bmN0aW9uXCIgJiYgKGdbU3ltYm9sLml0ZXJhdG9yXSA9IGZ1bmN0aW9uKCkgeyByZXR1cm4gdGhpczsgfSksIGc7XHJcbiAgICBmdW5jdGlvbiB2ZXJiKG4pIHsgcmV0dXJuIGZ1bmN0aW9uICh2KSB7IHJldHVybiBzdGVwKFtuLCB2XSk7IH07IH1cclxuICAgIGZ1bmN0aW9uIHN0ZXAob3ApIHtcclxuICAgICAgICBpZiAoZikgdGhyb3cgbmV3IFR5cGVFcnJvcihcIkdlbmVyYXRvciBpcyBhbHJlYWR5IGV4ZWN1dGluZy5cIik7XHJcbiAgICAgICAgd2hpbGUgKF8pIHRyeSB7XHJcbiAgICAgICAgICAgIGlmIChmID0gMSwgeSAmJiAodCA9IG9wWzBdICYgMiA/IHlbXCJyZXR1cm5cIl0gOiBvcFswXSA/IHlbXCJ0aHJvd1wiXSB8fCAoKHQgPSB5W1wicmV0dXJuXCJdKSAmJiB0LmNhbGwoeSksIDApIDogeS5uZXh0KSAmJiAhKHQgPSB0LmNhbGwoeSwgb3BbMV0pKS5kb25lKSByZXR1cm4gdDtcclxuICAgICAgICAgICAgaWYgKHkgPSAwLCB0KSBvcCA9IFtvcFswXSAmIDIsIHQudmFsdWVdO1xyXG4gICAgICAgICAgICBzd2l0Y2ggKG9wWzBdKSB7XHJcbiAgICAgICAgICAgICAgICBjYXNlIDA6IGNhc2UgMTogdCA9IG9wOyBicmVhaztcclxuICAgICAgICAgICAgICAgIGNhc2UgNDogXy5sYWJlbCsrOyByZXR1cm4geyB2YWx1ZTogb3BbMV0sIGRvbmU6IGZhbHNlIH07XHJcbiAgICAgICAgICAgICAgICBjYXNlIDU6IF8ubGFiZWwrKzsgeSA9IG9wWzFdOyBvcCA9IFswXTsgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICBjYXNlIDc6IG9wID0gXy5vcHMucG9wKCk7IF8udHJ5cy5wb3AoKTsgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICBkZWZhdWx0OlxyXG4gICAgICAgICAgICAgICAgICAgIGlmICghKHQgPSBfLnRyeXMsIHQgPSB0Lmxlbmd0aCA+IDAgJiYgdFt0Lmxlbmd0aCAtIDFdKSAmJiAob3BbMF0gPT09IDYgfHwgb3BbMF0gPT09IDIpKSB7IF8gPSAwOyBjb250aW51ZTsgfVxyXG4gICAgICAgICAgICAgICAgICAgIGlmIChvcFswXSA9PT0gMyAmJiAoIXQgfHwgKG9wWzFdID4gdFswXSAmJiBvcFsxXSA8IHRbM10pKSkgeyBfLmxhYmVsID0gb3BbMV07IGJyZWFrOyB9XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG9wWzBdID09PSA2ICYmIF8ubGFiZWwgPCB0WzFdKSB7IF8ubGFiZWwgPSB0WzFdOyB0ID0gb3A7IGJyZWFrOyB9XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHQgJiYgXy5sYWJlbCA8IHRbMl0pIHsgXy5sYWJlbCA9IHRbMl07IF8ub3BzLnB1c2gob3ApOyBicmVhazsgfVxyXG4gICAgICAgICAgICAgICAgICAgIGlmICh0WzJdKSBfLm9wcy5wb3AoKTtcclxuICAgICAgICAgICAgICAgICAgICBfLnRyeXMucG9wKCk7IGNvbnRpbnVlO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIG9wID0gYm9keS5jYWxsKHRoaXNBcmcsIF8pO1xyXG4gICAgICAgIH0gY2F0Y2ggKGUpIHsgb3AgPSBbNiwgZV07IHkgPSAwOyB9IGZpbmFsbHkgeyBmID0gdCA9IDA7IH1cclxuICAgICAgICBpZiAob3BbMF0gJiA1KSB0aHJvdyBvcFsxXTsgcmV0dXJuIHsgdmFsdWU6IG9wWzBdID8gb3BbMV0gOiB2b2lkIDAsIGRvbmU6IHRydWUgfTtcclxuICAgIH1cclxufVxyXG5cclxuZXhwb3J0IHZhciBfX2NyZWF0ZUJpbmRpbmcgPSBPYmplY3QuY3JlYXRlID8gKGZ1bmN0aW9uKG8sIG0sIGssIGsyKSB7XHJcbiAgICBpZiAoazIgPT09IHVuZGVmaW5lZCkgazIgPSBrO1xyXG4gICAgT2JqZWN0LmRlZmluZVByb3BlcnR5KG8sIGsyLCB7IGVudW1lcmFibGU6IHRydWUsIGdldDogZnVuY3Rpb24oKSB7IHJldHVybiBtW2tdOyB9IH0pO1xyXG59KSA6IChmdW5jdGlvbihvLCBtLCBrLCBrMikge1xyXG4gICAgaWYgKGsyID09PSB1bmRlZmluZWQpIGsyID0gaztcclxuICAgIG9bazJdID0gbVtrXTtcclxufSk7XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19leHBvcnRTdGFyKG0sIG8pIHtcclxuICAgIGZvciAodmFyIHAgaW4gbSkgaWYgKHAgIT09IFwiZGVmYXVsdFwiICYmICFPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwobywgcCkpIF9fY3JlYXRlQmluZGluZyhvLCBtLCBwKTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fdmFsdWVzKG8pIHtcclxuICAgIHZhciBzID0gdHlwZW9mIFN5bWJvbCA9PT0gXCJmdW5jdGlvblwiICYmIFN5bWJvbC5pdGVyYXRvciwgbSA9IHMgJiYgb1tzXSwgaSA9IDA7XHJcbiAgICBpZiAobSkgcmV0dXJuIG0uY2FsbChvKTtcclxuICAgIGlmIChvICYmIHR5cGVvZiBvLmxlbmd0aCA9PT0gXCJudW1iZXJcIikgcmV0dXJuIHtcclxuICAgICAgICBuZXh0OiBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgIGlmIChvICYmIGkgPj0gby5sZW5ndGgpIG8gPSB2b2lkIDA7XHJcbiAgICAgICAgICAgIHJldHVybiB7IHZhbHVlOiBvICYmIG9baSsrXSwgZG9uZTogIW8gfTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgdGhyb3cgbmV3IFR5cGVFcnJvcihzID8gXCJPYmplY3QgaXMgbm90IGl0ZXJhYmxlLlwiIDogXCJTeW1ib2wuaXRlcmF0b3IgaXMgbm90IGRlZmluZWQuXCIpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19yZWFkKG8sIG4pIHtcclxuICAgIHZhciBtID0gdHlwZW9mIFN5bWJvbCA9PT0gXCJmdW5jdGlvblwiICYmIG9bU3ltYm9sLml0ZXJhdG9yXTtcclxuICAgIGlmICghbSkgcmV0dXJuIG87XHJcbiAgICB2YXIgaSA9IG0uY2FsbChvKSwgciwgYXIgPSBbXSwgZTtcclxuICAgIHRyeSB7XHJcbiAgICAgICAgd2hpbGUgKChuID09PSB2b2lkIDAgfHwgbi0tID4gMCkgJiYgIShyID0gaS5uZXh0KCkpLmRvbmUpIGFyLnB1c2goci52YWx1ZSk7XHJcbiAgICB9XHJcbiAgICBjYXRjaCAoZXJyb3IpIHsgZSA9IHsgZXJyb3I6IGVycm9yIH07IH1cclxuICAgIGZpbmFsbHkge1xyXG4gICAgICAgIHRyeSB7XHJcbiAgICAgICAgICAgIGlmIChyICYmICFyLmRvbmUgJiYgKG0gPSBpW1wicmV0dXJuXCJdKSkgbS5jYWxsKGkpO1xyXG4gICAgICAgIH1cclxuICAgICAgICBmaW5hbGx5IHsgaWYgKGUpIHRocm93IGUuZXJyb3I7IH1cclxuICAgIH1cclxuICAgIHJldHVybiBhcjtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fc3ByZWFkKCkge1xyXG4gICAgZm9yICh2YXIgYXIgPSBbXSwgaSA9IDA7IGkgPCBhcmd1bWVudHMubGVuZ3RoOyBpKyspXHJcbiAgICAgICAgYXIgPSBhci5jb25jYXQoX19yZWFkKGFyZ3VtZW50c1tpXSkpO1xyXG4gICAgcmV0dXJuIGFyO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19zcHJlYWRBcnJheXMoKSB7XHJcbiAgICBmb3IgKHZhciBzID0gMCwgaSA9IDAsIGlsID0gYXJndW1lbnRzLmxlbmd0aDsgaSA8IGlsOyBpKyspIHMgKz0gYXJndW1lbnRzW2ldLmxlbmd0aDtcclxuICAgIGZvciAodmFyIHIgPSBBcnJheShzKSwgayA9IDAsIGkgPSAwOyBpIDwgaWw7IGkrKylcclxuICAgICAgICBmb3IgKHZhciBhID0gYXJndW1lbnRzW2ldLCBqID0gMCwgamwgPSBhLmxlbmd0aDsgaiA8IGpsOyBqKyssIGsrKylcclxuICAgICAgICAgICAgcltrXSA9IGFbal07XHJcbiAgICByZXR1cm4gcjtcclxufTtcclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2F3YWl0KHYpIHtcclxuICAgIHJldHVybiB0aGlzIGluc3RhbmNlb2YgX19hd2FpdCA/ICh0aGlzLnYgPSB2LCB0aGlzKSA6IG5ldyBfX2F3YWl0KHYpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hc3luY0dlbmVyYXRvcih0aGlzQXJnLCBfYXJndW1lbnRzLCBnZW5lcmF0b3IpIHtcclxuICAgIGlmICghU3ltYm9sLmFzeW5jSXRlcmF0b3IpIHRocm93IG5ldyBUeXBlRXJyb3IoXCJTeW1ib2wuYXN5bmNJdGVyYXRvciBpcyBub3QgZGVmaW5lZC5cIik7XHJcbiAgICB2YXIgZyA9IGdlbmVyYXRvci5hcHBseSh0aGlzQXJnLCBfYXJndW1lbnRzIHx8IFtdKSwgaSwgcSA9IFtdO1xyXG4gICAgcmV0dXJuIGkgPSB7fSwgdmVyYihcIm5leHRcIiksIHZlcmIoXCJ0aHJvd1wiKSwgdmVyYihcInJldHVyblwiKSwgaVtTeW1ib2wuYXN5bmNJdGVyYXRvcl0gPSBmdW5jdGlvbiAoKSB7IHJldHVybiB0aGlzOyB9LCBpO1xyXG4gICAgZnVuY3Rpb24gdmVyYihuKSB7IGlmIChnW25dKSBpW25dID0gZnVuY3Rpb24gKHYpIHsgcmV0dXJuIG5ldyBQcm9taXNlKGZ1bmN0aW9uIChhLCBiKSB7IHEucHVzaChbbiwgdiwgYSwgYl0pID4gMSB8fCByZXN1bWUobiwgdik7IH0pOyB9OyB9XHJcbiAgICBmdW5jdGlvbiByZXN1bWUobiwgdikgeyB0cnkgeyBzdGVwKGdbbl0odikpOyB9IGNhdGNoIChlKSB7IHNldHRsZShxWzBdWzNdLCBlKTsgfSB9XHJcbiAgICBmdW5jdGlvbiBzdGVwKHIpIHsgci52YWx1ZSBpbnN0YW5jZW9mIF9fYXdhaXQgPyBQcm9taXNlLnJlc29sdmUoci52YWx1ZS52KS50aGVuKGZ1bGZpbGwsIHJlamVjdCkgOiBzZXR0bGUocVswXVsyXSwgcik7IH1cclxuICAgIGZ1bmN0aW9uIGZ1bGZpbGwodmFsdWUpIHsgcmVzdW1lKFwibmV4dFwiLCB2YWx1ZSk7IH1cclxuICAgIGZ1bmN0aW9uIHJlamVjdCh2YWx1ZSkgeyByZXN1bWUoXCJ0aHJvd1wiLCB2YWx1ZSk7IH1cclxuICAgIGZ1bmN0aW9uIHNldHRsZShmLCB2KSB7IGlmIChmKHYpLCBxLnNoaWZ0KCksIHEubGVuZ3RoKSByZXN1bWUocVswXVswXSwgcVswXVsxXSk7IH1cclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fYXN5bmNEZWxlZ2F0b3Iobykge1xyXG4gICAgdmFyIGksIHA7XHJcbiAgICByZXR1cm4gaSA9IHt9LCB2ZXJiKFwibmV4dFwiKSwgdmVyYihcInRocm93XCIsIGZ1bmN0aW9uIChlKSB7IHRocm93IGU7IH0pLCB2ZXJiKFwicmV0dXJuXCIpLCBpW1N5bWJvbC5pdGVyYXRvcl0gPSBmdW5jdGlvbiAoKSB7IHJldHVybiB0aGlzOyB9LCBpO1xyXG4gICAgZnVuY3Rpb24gdmVyYihuLCBmKSB7IGlbbl0gPSBvW25dID8gZnVuY3Rpb24gKHYpIHsgcmV0dXJuIChwID0gIXApID8geyB2YWx1ZTogX19hd2FpdChvW25dKHYpKSwgZG9uZTogbiA9PT0gXCJyZXR1cm5cIiB9IDogZiA/IGYodikgOiB2OyB9IDogZjsgfVxyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hc3luY1ZhbHVlcyhvKSB7XHJcbiAgICBpZiAoIVN5bWJvbC5hc3luY0l0ZXJhdG9yKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiU3ltYm9sLmFzeW5jSXRlcmF0b3IgaXMgbm90IGRlZmluZWQuXCIpO1xyXG4gICAgdmFyIG0gPSBvW1N5bWJvbC5hc3luY0l0ZXJhdG9yXSwgaTtcclxuICAgIHJldHVybiBtID8gbS5jYWxsKG8pIDogKG8gPSB0eXBlb2YgX192YWx1ZXMgPT09IFwiZnVuY3Rpb25cIiA/IF9fdmFsdWVzKG8pIDogb1tTeW1ib2wuaXRlcmF0b3JdKCksIGkgPSB7fSwgdmVyYihcIm5leHRcIiksIHZlcmIoXCJ0aHJvd1wiKSwgdmVyYihcInJldHVyblwiKSwgaVtTeW1ib2wuYXN5bmNJdGVyYXRvcl0gPSBmdW5jdGlvbiAoKSB7IHJldHVybiB0aGlzOyB9LCBpKTtcclxuICAgIGZ1bmN0aW9uIHZlcmIobikgeyBpW25dID0gb1tuXSAmJiBmdW5jdGlvbiAodikgeyByZXR1cm4gbmV3IFByb21pc2UoZnVuY3Rpb24gKHJlc29sdmUsIHJlamVjdCkgeyB2ID0gb1tuXSh2KSwgc2V0dGxlKHJlc29sdmUsIHJlamVjdCwgdi5kb25lLCB2LnZhbHVlKTsgfSk7IH07IH1cclxuICAgIGZ1bmN0aW9uIHNldHRsZShyZXNvbHZlLCByZWplY3QsIGQsIHYpIHsgUHJvbWlzZS5yZXNvbHZlKHYpLnRoZW4oZnVuY3Rpb24odikgeyByZXNvbHZlKHsgdmFsdWU6IHYsIGRvbmU6IGQgfSk7IH0sIHJlamVjdCk7IH1cclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fbWFrZVRlbXBsYXRlT2JqZWN0KGNvb2tlZCwgcmF3KSB7XHJcbiAgICBpZiAoT2JqZWN0LmRlZmluZVByb3BlcnR5KSB7IE9iamVjdC5kZWZpbmVQcm9wZXJ0eShjb29rZWQsIFwicmF3XCIsIHsgdmFsdWU6IHJhdyB9KTsgfSBlbHNlIHsgY29va2VkLnJhdyA9IHJhdzsgfVxyXG4gICAgcmV0dXJuIGNvb2tlZDtcclxufTtcclxuXHJcbnZhciBfX3NldE1vZHVsZURlZmF1bHQgPSBPYmplY3QuY3JlYXRlID8gKGZ1bmN0aW9uKG8sIHYpIHtcclxuICAgIE9iamVjdC5kZWZpbmVQcm9wZXJ0eShvLCBcImRlZmF1bHRcIiwgeyBlbnVtZXJhYmxlOiB0cnVlLCB2YWx1ZTogdiB9KTtcclxufSkgOiBmdW5jdGlvbihvLCB2KSB7XHJcbiAgICBvW1wiZGVmYXVsdFwiXSA9IHY7XHJcbn07XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19pbXBvcnRTdGFyKG1vZCkge1xyXG4gICAgaWYgKG1vZCAmJiBtb2QuX19lc01vZHVsZSkgcmV0dXJuIG1vZDtcclxuICAgIHZhciByZXN1bHQgPSB7fTtcclxuICAgIGlmIChtb2QgIT0gbnVsbCkgZm9yICh2YXIgayBpbiBtb2QpIGlmIChrICE9PSBcImRlZmF1bHRcIiAmJiBPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwobW9kLCBrKSkgX19jcmVhdGVCaW5kaW5nKHJlc3VsdCwgbW9kLCBrKTtcclxuICAgIF9fc2V0TW9kdWxlRGVmYXVsdChyZXN1bHQsIG1vZCk7XHJcbiAgICByZXR1cm4gcmVzdWx0O1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19pbXBvcnREZWZhdWx0KG1vZCkge1xyXG4gICAgcmV0dXJuIChtb2QgJiYgbW9kLl9fZXNNb2R1bGUpID8gbW9kIDogeyBkZWZhdWx0OiBtb2QgfTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fY2xhc3NQcml2YXRlRmllbGRHZXQocmVjZWl2ZXIsIHByaXZhdGVNYXApIHtcclxuICAgIGlmICghcHJpdmF0ZU1hcC5oYXMocmVjZWl2ZXIpKSB7XHJcbiAgICAgICAgdGhyb3cgbmV3IFR5cGVFcnJvcihcImF0dGVtcHRlZCB0byBnZXQgcHJpdmF0ZSBmaWVsZCBvbiBub24taW5zdGFuY2VcIik7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gcHJpdmF0ZU1hcC5nZXQocmVjZWl2ZXIpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19jbGFzc1ByaXZhdGVGaWVsZFNldChyZWNlaXZlciwgcHJpdmF0ZU1hcCwgdmFsdWUpIHtcclxuICAgIGlmICghcHJpdmF0ZU1hcC5oYXMocmVjZWl2ZXIpKSB7XHJcbiAgICAgICAgdGhyb3cgbmV3IFR5cGVFcnJvcihcImF0dGVtcHRlZCB0byBzZXQgcHJpdmF0ZSBmaWVsZCBvbiBub24taW5zdGFuY2VcIik7XHJcbiAgICB9XHJcbiAgICBwcml2YXRlTWFwLnNldChyZWNlaXZlciwgdmFsdWUpO1xyXG4gICAgcmV0dXJuIHZhbHVlO1xyXG59XHJcbiIsImV4cG9ydCBjbGFzcyBMYXRleEVudmlyb25tZW50c1NldHRpbmdzIHtcbiAgcHVibGljIGRlZmF1bHRFbnZpcm9ubWVudCA9ICdtdWx0bGluZSc7XG4gIHB1YmxpYyBjdXN0b21FbnZpcm9ubWVudHM6IHN0cmluZ1tdID0gW107XG59XG5cbmV4cG9ydCB0eXBlIExvYWRlZFNldHRpbmdzID0gUGFydGlhbDxMYXRleEVudmlyb25tZW50c1NldHRpbmdzPjtcblxuZXhwb3J0IGZ1bmN0aW9uIGVuc3VyZVNldHRpbmdzKFxuICBsb2FkZWQ6IExvYWRlZFNldHRpbmdzLFxuKTogTGF0ZXhFbnZpcm9ubWVudHNTZXR0aW5ncyB7XG4gIGNvbnN0IHNldHRpbmdzID0gbmV3IExhdGV4RW52aXJvbm1lbnRzU2V0dGluZ3MoKTtcblxuICBzZXR0aW5ncy5kZWZhdWx0RW52aXJvbm1lbnQgPVxuICAgIGxvYWRlZC5kZWZhdWx0RW52aXJvbm1lbnQgPz8gc2V0dGluZ3MuZGVmYXVsdEVudmlyb25tZW50O1xuXG4gIHNldHRpbmdzLmN1c3RvbUVudmlyb25tZW50cyA9XG4gICAgbG9hZGVkLmN1c3RvbUVudmlyb25tZW50cyA/PyBzZXR0aW5ncy5jdXN0b21FbnZpcm9ubWVudHM7XG5cbiAgcmV0dXJuIHNldHRpbmdzO1xufVxuIiwiaW1wb3J0IENvZGVNaXJyb3IgZnJvbSAnY29kZW1pcnJvcic7XG5cbmV4cG9ydCBpbnRlcmZhY2UgUG9zUmFuZ2Uge1xuICBmcm9tOiBDb2RlTWlycm9yLlBvc2l0aW9uO1xuICB0bzogQ29kZU1pcnJvci5Qb3NpdGlvbjtcbn1cblxuY29uc3QgQkVHSU5fTEVOR1RIID0gODtcbmNvbnN0IEVORF9MRU5HVEggPSA2O1xuXG5leHBvcnQgY2xhc3MgRW52aXJvbm1lbnQge1xuICBjb25zdHJ1Y3RvcihcbiAgICBwdWJsaWMgZG9jOiBDb2RlTWlycm9yLkRvYyxcbiAgICBwcml2YXRlIF9uYW1lOiBzdHJpbmcsXG4gICAgcHJpdmF0ZSBfc3RhcnQ6IFBvc1JhbmdlLFxuICAgIHByaXZhdGUgX2VuZDogUG9zUmFuZ2UsXG4gICkge31cblxuICBwdWJsaWMgZ2V0IG5hbWUoKTogc3RyaW5nIHtcbiAgICByZXR1cm4gdGhpcy5fbmFtZTtcbiAgfVxuXG4gIHB1YmxpYyBnZXQgc3RhcnQoKTogUG9zUmFuZ2Uge1xuICAgIHJldHVybiB0aGlzLl9zdGFydDtcbiAgfVxuXG4gIHB1YmxpYyBnZXQgZW5kKCk6IFBvc1JhbmdlIHtcbiAgICByZXR1cm4gdGhpcy5fZW5kO1xuICB9XG5cbiAgcHVibGljIGdldCBiZWdpblN0cmluZygpOiBzdHJpbmcge1xuICAgIHJldHVybiBgXFxcXGJlZ2lueyR7dGhpcy5fbmFtZX19YDtcbiAgfVxuXG4gIHB1YmxpYyBnZXQgZW5kU3RyaW5nKCk6IHN0cmluZyB7XG4gICAgcmV0dXJuIGBcXFxcZW5keyR7dGhpcy5fbmFtZX19YDtcbiAgfVxuXG4gIHB1YmxpYyByZXBsYWNlKGVudk5hbWU6IHN0cmluZyk6IEVudmlyb25tZW50IHtcbiAgICB0aGlzLl9uYW1lID0gZW52TmFtZTtcbiAgICB0aGlzLmRvYy5yZXBsYWNlUmFuZ2UodGhpcy5iZWdpblN0cmluZywgdGhpcy5zdGFydC5mcm9tLCB0aGlzLnN0YXJ0LnRvKTtcbiAgICB0aGlzLmRvYy5yZXBsYWNlUmFuZ2UodGhpcy5lbmRTdHJpbmcsIHRoaXMuZW5kLmZyb20sIHRoaXMuZW5kLnRvKTtcbiAgICB0aGlzLl9zdGFydC50byA9IHtcbiAgICAgIGxpbmU6IHRoaXMuc3RhcnQuZnJvbS5saW5lLFxuICAgICAgY2g6IHRoaXMuc3RhcnQuZnJvbS5jaCArIHRoaXMuYmVnaW5TdHJpbmcubGVuZ3RoLFxuICAgIH07XG4gICAgdGhpcy5fZW5kLnRvID0ge1xuICAgICAgbGluZTogdGhpcy5lbmQuZnJvbS5saW5lLFxuICAgICAgY2g6IHRoaXMuZW5kLmZyb20uY2ggKyB0aGlzLmVuZFN0cmluZy5sZW5ndGgsXG4gICAgfTtcbiAgICByZXR1cm4gdGhpcztcbiAgfVxuXG4gIHB1YmxpYyBwcmludChjb250ZW50cyA9ICdcXG5cXG4nKTogc3RyaW5nIHtcbiAgICByZXR1cm4gYCR7dGhpcy5iZWdpblN0cmluZ30ke2NvbnRlbnRzfSR7dGhpcy5lbmRTdHJpbmd9YDtcbiAgfVxuXG4gIHByaXZhdGUgc3RhdGljIG5ld1JhbmdlKFxuICAgIGN1cnNvcjogQ29kZU1pcnJvci5Qb3NpdGlvbixcbiAgICBlbnZOYW1lOiBzdHJpbmcsXG4gICAgbGluZU9mZnNldDogbnVtYmVyLFxuICAgIGNoT2Zmc2V0OiBudW1iZXIsXG4gICk6IFBvc1JhbmdlIHtcbiAgICByZXR1cm4ge1xuICAgICAgZnJvbToge1xuICAgICAgICBsaW5lOiBjdXJzb3IubGluZSArIGxpbmVPZmZzZXQsXG4gICAgICAgIGNoOiBjdXJzb3IuY2gsXG4gICAgICB9LFxuICAgICAgdG86IHtcbiAgICAgICAgbGluZTogY3Vyc29yLmxpbmUgKyBsaW5lT2Zmc2V0LFxuICAgICAgICBjaDogY3Vyc29yLmNoICsgY2hPZmZzZXQgKyBlbnZOYW1lLmxlbmd0aCxcbiAgICAgIH0sXG4gICAgfTtcbiAgfVxuXG4gIHB1YmxpYyBzdGF0aWMgY3JlYXRlKFxuICAgIGVudk5hbWU6IHN0cmluZyxcbiAgICBkb2M6IENvZGVNaXJyb3IuRG9jLFxuICAgIGN1cnNvcjogQ29kZU1pcnJvci5Qb3NpdGlvbixcbiAgKTogRW52aXJvbm1lbnQge1xuICAgIGNvbnN0IG5ld0xpbmUgPSBuZXh0TGluZShjdXJzb3IsIHRydWUpO1xuICAgIGNvbnN0IG5ld0Vudmlyb25tZW50ID0gbmV3IEVudmlyb25tZW50KFxuICAgICAgZG9jLFxuICAgICAgZW52TmFtZSxcbiAgICAgIHRoaXMubmV3UmFuZ2UobmV3TGluZSwgZW52TmFtZSwgMCwgQkVHSU5fTEVOR1RIKSxcbiAgICAgIHRoaXMubmV3UmFuZ2UobmV3TGluZSwgZW52TmFtZSwgMiwgRU5EX0xFTkdUSCksXG4gICAgKTtcblxuICAgIGNvbnN0IGxpbmUgPSBkb2MuZ2V0TGluZShjdXJzb3IubGluZSk7XG4gICAgY29uc3QgZnJvbnRQYWQgPSBnZXRQYWQobGluZS5zdWJzdHIoMCwgY3Vyc29yLmNoKSk7XG4gICAgY29uc3QgcmVhclBhZCA9IGdldFBhZChsaW5lLnN1YnN0cihjdXJzb3IuY2gpKTtcblxuICAgIGRvYy5yZXBsYWNlUmFuZ2UoZnJvbnRQYWQgKyBuZXdFbnZpcm9ubWVudC5wcmludCgpICsgcmVhclBhZCwgY3Vyc29yKTtcbiAgICBkb2Muc2V0Q3Vyc29yKG5leHRMaW5lKG5ld0xpbmUsIGZhbHNlLCBmcm9udFBhZC5sZW5ndGgpKTtcbiAgICByZXR1cm4gbmV3RW52aXJvbm1lbnQ7XG4gIH1cblxuICBwdWJsaWMgc3RhdGljIHdyYXAoXG4gICAgZW52TmFtZTogc3RyaW5nLFxuICAgIGRvYzogQ29kZU1pcnJvci5Eb2MsXG4gICAgZnJvbTogQ29kZU1pcnJvci5Qb3NpdGlvbixcbiAgICB0bzogQ29kZU1pcnJvci5Qb3NpdGlvbixcbiAgICBvdXRlclBhZCA9ICdcXG4nLFxuICApOiBFbnZpcm9ubWVudCB7XG4gICAgY29uc3QgbmV3RW52aXJvbm1lbnQgPSBuZXcgRW52aXJvbm1lbnQoXG4gICAgICBkb2MsXG4gICAgICBlbnZOYW1lLFxuICAgICAgdGhpcy5uZXdSYW5nZShmcm9tLCBlbnZOYW1lLCAwLCBCRUdJTl9MRU5HVEgpLFxuICAgICAgdGhpcy5uZXdSYW5nZShuZXh0TGluZSh0bywgdHJ1ZSwgMiksIGVudk5hbWUsIDIsIEVORF9MRU5HVEgpLFxuICAgICk7XG4gICAgZG9jLnJlcGxhY2VSYW5nZShvdXRlclBhZCArIG5ld0Vudmlyb25tZW50LmVuZFN0cmluZywgdG8pO1xuICAgIGRvYy5yZXBsYWNlUmFuZ2UobmV3RW52aXJvbm1lbnQuYmVnaW5TdHJpbmcgKyBvdXRlclBhZCwgZnJvbSk7XG4gICAgaWYgKGRvYy5zb21ldGhpbmdTZWxlY3RlZCgpKSB7XG4gICAgICBkb2Muc2V0Q3Vyc29yKG5leHRMaW5lKGZyb20sIHRydWUpKTtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgbGluZU9mZnNldCA9IG5ld0Vudmlyb25tZW50LnN0YXJ0LmZyb20ubGluZSAtIGZyb20ubGluZTtcbiAgICAgIGRvYy5zZXRDdXJzb3IobmV4dExpbmUoZG9jLmdldEN1cnNvcigpLCBmYWxzZSwgbGluZU9mZnNldCkpO1xuICAgIH1cblxuICAgIHJldHVybiBuZXdFbnZpcm9ubWVudDtcbiAgfVxuXG4gIHB1YmxpYyB1bndyYXAoKTogdm9pZCB7XG4gICAgdGhpcy5kb2MucmVwbGFjZVJhbmdlKCcnLCB0aGlzLnN0YXJ0LmZyb20sIHRoaXMuc3RhcnQudG8pO1xuICAgIHRoaXMuZG9jLnJlcGxhY2VSYW5nZSgnJywgdGhpcy5lbmQuZnJvbSwgdGhpcy5lbmQudG8pO1xuICB9XG59XG5cbmZ1bmN0aW9uIG5leHRMaW5lKFxuICBjdXJzb3I6IENvZGVNaXJyb3IuUG9zaXRpb24sXG4gIGNyID0gZmFsc2UsXG4gIG9mZnNldCA9IDEsXG4pOiBDb2RlTWlycm9yLlBvc2l0aW9uIHtcbiAgcmV0dXJuIHsgbGluZTogY3Vyc29yLmxpbmUgKyBvZmZzZXQsIGNoOiBjciA/IDAgOiBjdXJzb3IuY2ggfTtcbn1cblxuZnVuY3Rpb24gZ2V0UGFkKHRleHQ6IHN0cmluZyk6IHN0cmluZyB7XG4gIGlmICh0ZXh0Lm1hdGNoKC9eWyBcXHRdKiQvKSAhPT0gbnVsbCkge1xuICAgIHJldHVybiAnJztcbiAgfVxuICByZXR1cm4gJ1xcbic7XG59XG4iLCJpbXBvcnQgQ29kZU1pcnJvciBmcm9tICdjb2RlbWlycm9yJztcbmltcG9ydCB7IEVudmlyb25tZW50LCBQb3NSYW5nZSB9IGZyb20gJy4vZW52aXJvbm1lbnQnO1xuXG5leHBvcnQgY2xhc3MgTWF0aEJsb2NrIHtcbiAgcmVhZG9ubHkgc3RhcnRQb3NpdGlvbjogQ29kZU1pcnJvci5Qb3NpdGlvbjtcbiAgcmVhZG9ubHkgZW5kUG9zaXRpb246IENvZGVNaXJyb3IuUG9zaXRpb247XG4gIHB1YmxpYyBkb2M6IENvZGVNaXJyb3IuRG9jO1xuXG4gIGNvbnN0cnVjdG9yKGRvYzogQ29kZU1pcnJvci5Eb2MsIGN1cnNvcjogQ29kZU1pcnJvci5Qb3NpdGlvbikge1xuICAgIGNvbnN0IHNlYXJjaEN1cnNvciA9IGRvYy5nZXRTZWFyY2hDdXJzb3IoJyQkJywgY3Vyc29yKTtcbiAgICB0aGlzLnN0YXJ0UG9zaXRpb24gPVxuICAgICAgc2VhcmNoQ3Vyc29yLmZpbmRQcmV2aW91cygpICE9PSBmYWxzZVxuICAgICAgICA/IHNlYXJjaEN1cnNvci50bygpXG4gICAgICAgIDogeyBsaW5lOiBkb2MuZmlyc3RMaW5lKCksIGNoOiAwIH07XG4gICAgdGhpcy5lbmRQb3NpdGlvbiA9XG4gICAgICBzZWFyY2hDdXJzb3IuZmluZE5leHQoKSAhPT0gZmFsc2VcbiAgICAgICAgPyBzZWFyY2hDdXJzb3IuZnJvbSgpXG4gICAgICAgIDogeyBsaW5lOiBkb2MubGFzdExpbmUoKSwgY2g6IGRvYy5nZXRMaW5lKGRvYy5sYXN0TGluZSgpKS5sZW5ndGggLSAxIH07XG4gICAgdGhpcy5kb2MgPSBkb2M7XG4gIH1cblxuICBwdWJsaWMgZ2V0RW5jbG9zaW5nRW52aXJvbm1lbnQoXG4gICAgY3Vyc29yOiBDb2RlTWlycm9yLlBvc2l0aW9uLFxuICApOiBFbnZpcm9ubWVudCB8IHVuZGVmaW5lZCB7XG4gICAgY29uc3QgYmVnaW5FbmRzID0gbmV3IEJlZ2luRW5kcyhcbiAgICAgIHRoaXMuZG9jLFxuICAgICAgdGhpcy5zdGFydFBvc2l0aW9uLFxuICAgICAgdGhpcy5lbmRQb3NpdGlvbixcbiAgICApO1xuICAgIGNvbnN0IGVudmlyb25tZW50cyA9IEFycmF5LmZyb20oYmVnaW5FbmRzKTtcblxuICAgIGlmIChiZWdpbkVuZHMuaXNPcGVuKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ3VuY2xvc2VkIGVudmlyb25tZW50cyBpbiBibG9jaycpO1xuICAgIH1cbiAgICBjb25zdCBzdGFydCA9IGVudmlyb25tZW50c1xuICAgICAgLmZpbHRlcigoZW52KSA9PiB7XG4gICAgICAgIGNvbnN0IGZyb20gPSBlbnYucG9zLmZyb207XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgZW52LnR5cGUgPT09ICdiZWdpbicgJiZcbiAgICAgICAgICAoZnJvbS5saW5lIDwgY3Vyc29yLmxpbmUgfHxcbiAgICAgICAgICAgIChmcm9tLmxpbmUgPT09IGN1cnNvci5saW5lICYmIGZyb20uY2ggPD0gY3Vyc29yLmNoKSlcbiAgICAgICAgKTtcbiAgICAgIH0pXG4gICAgICAucG9wKCk7XG5cbiAgICBpZiAoc3RhcnQgPT09IHVuZGVmaW5lZCkge1xuICAgICAgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICB9XG5cbiAgICBjb25zdCBzdGFydFRvID0gc3RhcnQucG9zLnRvO1xuICAgIGNvbnN0IGFmdGVyID0gZW52aXJvbm1lbnRzLmZpbHRlcigoZW52KSA9PiB7XG4gICAgICBjb25zdCBmcm9tID0gZW52LnBvcy5mcm9tO1xuICAgICAgcmV0dXJuIChcbiAgICAgICAgZnJvbS5saW5lID4gc3RhcnRUby5saW5lIHx8XG4gICAgICAgIChmcm9tLmxpbmUgPT09IHN0YXJ0VG8ubGluZSAmJiBmcm9tLmNoID4gc3RhcnRUby5jaClcbiAgICAgICk7XG4gICAgfSk7XG5cbiAgICBsZXQgb3BlbiA9IDE7XG4gICAgbGV0IGVuZDogQmVnaW5FbmQgfCB1bmRlZmluZWQ7XG4gICAgZm9yIChjb25zdCBlbnYgb2YgYWZ0ZXIpIHtcbiAgICAgIGlmIChlbnYudHlwZSA9PT0gJ2JlZ2luJykge1xuICAgICAgICBvcGVuKys7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBvcGVuLS07XG4gICAgICAgIGlmIChvcGVuID09PSAwKSB7XG4gICAgICAgICAgZW5kID0gZW52O1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgaWYgKGVuZCA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ2N1cnJlbnQgZW52aXJvbm1lbnQgaXMgbmV2ZXIgY2xvc2VkJyk7XG4gICAgfVxuXG4gICAgY29uc3QgZW5kVG8gPSBlbmQucG9zLnRvO1xuICAgIGlmIChcbiAgICAgIGVuZFRvLmxpbmUgPCBjdXJzb3IubGluZSB8fFxuICAgICAgKGVuZFRvLmxpbmUgPT09IGN1cnNvci5saW5lICYmIGVuZFRvLmNoIDwgY3Vyc29yLmNoKVxuICAgICkge1xuICAgICAgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICB9XG5cbiAgICByZXR1cm4gbmV3IEVudmlyb25tZW50KHRoaXMuZG9jLCBzdGFydC5uYW1lLCBzdGFydC5wb3MsIGVuZC5wb3MpO1xuICB9XG5cbiAgcHVibGljIHN0YXRpYyBpc01hdGhNb2RlKFxuICAgIGN1cnNvcjogQ29kZU1pcnJvci5Qb3NpdGlvbixcbiAgICBlZGl0b3I6IENvZGVNaXJyb3IuRWRpdG9yLFxuICApOiBib29sZWFuIHtcbiAgICBjb25zdCB0b2tlbiA9IGVkaXRvci5nZXRUb2tlbkF0KGN1cnNvcik7XG4gICAgY29uc3Qgc3RhdGUgPSB0b2tlbi5zdGF0ZTtcbiAgICByZXR1cm4gc3RhdGUuaG1kSW5uZXJTdHlsZSA9PT0gJ21hdGgnO1xuICB9XG59XG5cbmludGVyZmFjZSBCZWdpbkVuZCB7XG4gIG5hbWU6IHN0cmluZztcbiAgdHlwZTogJ2JlZ2luJyB8ICdlbmQnO1xuICBwb3M6IFBvc1JhbmdlO1xufVxuXG5jbGFzcyBCZWdpbkVuZHMgaW1wbGVtZW50cyBJdGVyYWJsZUl0ZXJhdG9yPEJlZ2luRW5kPiB7XG4gIHByaXZhdGUgcmVhZG9ubHkgb3BlbkVudnM6IEJlZ2luRW5kW10gPSBbXTtcbiAgcHJpdmF0ZSBzZWFyY2g6IENvZGVNaXJyb3IuU2VhcmNoQ3Vyc29yO1xuICBjb25zdHJ1Y3RvcihcbiAgICByZWFkb25seSBkb2M6IENvZGVNaXJyb3IuRG9jLFxuICAgIHJlYWRvbmx5IHN0YXJ0OiBDb2RlTWlycm9yLlBvc2l0aW9uLFxuICAgIHJlYWRvbmx5IGVuZDogQ29kZU1pcnJvci5Qb3NpdGlvbixcbiAgKSB7XG4gICAgdGhpcy5zZWFyY2ggPSB0aGlzLmdldEVudkN1cnNvcih0aGlzLnN0YXJ0KTtcbiAgfVxuXG4gIHB1YmxpYyByZXNldCgpOiB2b2lkIHtcbiAgICB0aGlzLnNlYXJjaCA9IHRoaXMuZ2V0RW52Q3Vyc29yKHRoaXMuc3RhcnQpO1xuICB9XG5cbiAgcHJpdmF0ZSBnZXRFbnZDdXJzb3Ioc3RhcnQ6IENvZGVNaXJyb3IuUG9zaXRpb24pOiBDb2RlTWlycm9yLlNlYXJjaEN1cnNvciB7XG4gICAgcmV0dXJuIHRoaXMuZG9jLmdldFNlYXJjaEN1cnNvcigvXFxcXChiZWdpbnxlbmQpe1xccyooW159XSspXFxzKn0vbSwgc3RhcnQpO1xuICB9XG5cbiAgcHVibGljIGdldCBpc09wZW4oKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIHRoaXMub3BlbkVudnMubGVuZ3RoID4gMDtcbiAgfVxuXG4gIFtTeW1ib2wuaXRlcmF0b3JdKCk6IEl0ZXJhYmxlSXRlcmF0b3I8QmVnaW5FbmQ+IHtcbiAgICB0aGlzLnJlc2V0KCk7XG4gICAgcmV0dXJuIHRoaXM7XG4gIH1cblxuICBuZXh0KCk6IEl0ZXJhdG9yUmVzdWx0PEJlZ2luRW5kPiB7XG4gICAgY29uc3QgbWF0Y2ggPSB0aGlzLnNlYXJjaC5maW5kTmV4dCgpO1xuICAgIGNvbnN0IHRvID0gdGhpcy5zZWFyY2gudG8oKTtcblxuICAgIGlmIChcbiAgICAgIG1hdGNoID09PSB0cnVlIHx8XG4gICAgICBtYXRjaCA9PT0gZmFsc2UgfHxcbiAgICAgIHRvLmxpbmUgPiB0aGlzLmVuZC5saW5lIHx8XG4gICAgICAodG8ubGluZSA9PT0gdGhpcy5lbmQubGluZSAmJiB0by5jaCA+IHRoaXMuZW5kLmNoKVxuICAgICkge1xuICAgICAgcmV0dXJuIHsgZG9uZTogdHJ1ZSwgdmFsdWU6IG51bGwgfTtcbiAgICB9XG5cbiAgICBzd2l0Y2ggKG1hdGNoWzFdKSB7XG4gICAgICBjYXNlICdiZWdpbic6IHtcbiAgICAgICAgY29uc3QgY3VycmVudDogQmVnaW5FbmQgPSB7XG4gICAgICAgICAgbmFtZTogbWF0Y2hbMl0sXG4gICAgICAgICAgdHlwZTogJ2JlZ2luJyxcbiAgICAgICAgICBwb3M6IHtcbiAgICAgICAgICAgIGZyb206IHRoaXMuc2VhcmNoLmZyb20oKSxcbiAgICAgICAgICAgIHRvOiB0aGlzLnNlYXJjaC50bygpLFxuICAgICAgICAgIH0sXG4gICAgICAgIH07XG4gICAgICAgIHRoaXMub3BlbkVudnMucHVzaChjdXJyZW50KTtcbiAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICBkb25lOiBmYWxzZSxcbiAgICAgICAgICB2YWx1ZTogY3VycmVudCxcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICAgIGNhc2UgJ2VuZCc6IHtcbiAgICAgICAgY29uc3QgY3VycmVudCA9IHRoaXMub3BlbkVudnMucG9wKCk7XG4gICAgICAgIGlmIChjdXJyZW50ID09PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ2Nsb3NpbmcgZW52aXJvbm1lbnQgd2hpY2ggd2FzIG5ldmVyIG9wZW5lZCcpO1xuICAgICAgICB9XG4gICAgICAgIGlmIChjdXJyZW50Lm5hbWUgIT09IG1hdGNoWzJdKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdlbnZpcm9ubWVudCBub3QgY2xvc2VkIHByb3Blcmx5Jyk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICBkb25lOiBmYWxzZSxcbiAgICAgICAgICB2YWx1ZToge1xuICAgICAgICAgICAgbmFtZTogbWF0Y2hbMl0sXG4gICAgICAgICAgICB0eXBlOiAnZW5kJyxcbiAgICAgICAgICAgIHBvczoge1xuICAgICAgICAgICAgICBmcm9tOiB0aGlzLnNlYXJjaC5mcm9tKCksXG4gICAgICAgICAgICAgIHRvOiB0aGlzLnNlYXJjaC50bygpLFxuICAgICAgICAgICAgfSxcbiAgICAgICAgICB9LFxuICAgICAgICB9O1xuICAgICAgfVxuICAgIH1cbiAgICB0aHJvdyBuZXcgRXJyb3IoYHJlZ2V4IHJldHVybmVkIHVuZXhwZWN0ZWQgcmVzdWx0ICR7bWF0Y2hbMV0gYXMgc3RyaW5nfWApO1xuICB9XG59XG4iLCJjb25zdCBESVNQTEFZX0VRVUFUSU9OUyA9IFtcbiAgJ2VxdWF0aW9uJyxcbiAgJ2VxdWF0aW9uKicsXG4gICdnYXRoZXInLFxuICAnZ2F0aGVyKicsXG4gICdtdWx0bGluZScsXG4gICdtdWx0bGluZSonLFxuICAnc3BsaXQnLFxuICAnYWxpZ24nLFxuICAnYWxpZ24qJyxcbiAgJ2ZsYWxpZ24nLFxuICAnZmxhbGlnbionLFxuICAnYWxpZ25hdCcsXG4gICdhbGlnbmF0KicsXG5dO1xuXG5jb25zdCBNQVRSSUNFUyA9IFtcbiAgJ21hdHJpeCcsXG4gICdwbWF0cml4JyxcbiAgJ2JtYXRyaXgnLFxuICAnQm1hdHJpeCcsXG4gICd2bWF0cml4JyxcbiAgJ1ZtYXRyaXgnLFxuICAnc21hbGxtYXRyaXgnLFxuXTtcblxuY29uc3QgU1VCX0VOVklST05NRU5UUyA9IFsnbXVsdGxpbmVkJywgJ2dhdGhlcmVkJywgJ2FsaWduZWQnLCAnY2FzZXMnXTtcblxuZXhwb3J0IGNvbnN0IERFRkFVTFRfRU5WSVJPTk1FTlRTID0gW1xuICAuLi5ESVNQTEFZX0VRVUFUSU9OUyxcbiAgLi4uTUFUUklDRVMsXG4gIC4uLlNVQl9FTlZJUk9OTUVOVFMsXG5dO1xuIiwiaW1wb3J0IHsgQXBwLCBGdXp6eVN1Z2dlc3RNb2RhbCB9IGZyb20gJ29ic2lkaWFuJztcbmltcG9ydCB7IExhdGV4RW52aXJvbm1lbnRzU2V0dGluZ3MgfSBmcm9tICcuL3NldHRpbmdzJztcbmltcG9ydCB7IERFRkFVTFRfRU5WSVJPTk1FTlRTIH0gZnJvbSAnLi9lbnZpcm9ubWVudE5hbWVzJztcblxuZXhwb3J0IGNsYXNzIEVudk1vZGFsIGV4dGVuZHMgRnV6enlTdWdnZXN0TW9kYWw8c3RyaW5nPiB7XG4gIHByaXZhdGUgbWF0Y2hlZDogYm9vbGVhbiA9IGZhbHNlO1xuICBjb25zdHJ1Y3RvcihcbiAgICBhcHA6IEFwcCxcbiAgICBwcml2YXRlIHJlYWRvbmx5IHNldHRpbmdzOiBMYXRleEVudmlyb25tZW50c1NldHRpbmdzLFxuICAgIHByaXZhdGUgcmVhZG9ubHkgbmFtZTogc3RyaW5nLFxuICAgIHByaXZhdGUgcmVhZG9ubHkgY2FsbGJhY2s6IChuYW1lOiBzdHJpbmcpID0+IHZvaWQsXG4gICkge1xuICAgIHN1cGVyKGFwcCk7XG4gICAgdGhpcy5zZXRJbnN0cnVjdGlvbnMoW1xuICAgICAgeyBjb21tYW5kOiAn4oaR4oaTJywgcHVycG9zZTogJ3RvIG5hdmlnYXRlJyB9LFxuICAgICAgeyBjb21tYW5kOiAn4oa1JywgcHVycG9zZTogJ3RvIHNlbGVjdCcgfSxcbiAgICAgIHsgY29tbWFuZDogJ2VzYycsIHB1cnBvc2U6ICd0byBkaXNtaXNzJyB9LFxuICAgIF0pO1xuICAgIHRoaXMuc2V0UGxhY2Vob2xkZXIoJ2Vudmlyb25tZW50IG5hbWUnKTtcbiAgfVxuXG4gIHB1YmxpYyBnZXRJdGVtcygpOiBzdHJpbmdbXSB7XG4gICAgcmV0dXJuIEFycmF5LmZyb20oXG4gICAgICBuZXcgU2V0KFxuICAgICAgICBbdGhpcy5zZXR0aW5ncy5kZWZhdWx0RW52aXJvbm1lbnRdLmNvbmNhdChcbiAgICAgICAgICB0aGlzLnNldHRpbmdzLmN1c3RvbUVudmlyb25tZW50cyxcbiAgICAgICAgICBERUZBVUxUX0VOVklST05NRU5UUyxcbiAgICAgICAgKSxcbiAgICAgICksXG4gICAgKTtcbiAgfVxuXG4gIHB1YmxpYyBnZXRJdGVtVGV4dChpdGVtOiBzdHJpbmcpOiBzdHJpbmcge1xuICAgIHRoaXMubWF0Y2hlZCA9IHRydWU7XG4gICAgcmV0dXJuIGl0ZW07XG4gIH1cblxuICBwdWJsaWMgb25Ob1N1Z2dlc3Rpb24oKTogdm9pZCB7XG4gICAgdGhpcy5tYXRjaGVkID0gZmFsc2U7XG4gIH1cblxuICBwdWJsaWMgb25DaG9vc2VJdGVtKGl0ZW06IHN0cmluZywgX2V2dDogTW91c2VFdmVudCB8IEtleWJvYXJkRXZlbnQpOiB2b2lkIHtcbiAgICBpZiAodGhpcy5tYXRjaGVkKSB7XG4gICAgICB0aGlzLmNhbGxiYWNrKGl0ZW0pO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmNhbGxiYWNrKHRoaXMuaW5wdXRFbC52YWx1ZSk7XG4gICAgfVxuICB9XG5cbiAgc3RhdGljIGNhbGxiYWNrKFxuICAgIGFwcDogQXBwLFxuICAgIHNldHRpbmdzOiBMYXRleEVudmlyb25tZW50c1NldHRpbmdzLFxuICAgIGRlZmF1bHROYW1lOiBzdHJpbmcsXG4gICAgY2FsbDogKG5hbWU6IHN0cmluZykgPT4gdm9pZCxcbiAgKTogdm9pZCB7XG4gICAgbmV3IEVudk1vZGFsKGFwcCwgc2V0dGluZ3MsIGRlZmF1bHROYW1lLCBjYWxsKS5vcGVuKCk7XG4gIH1cbn1cbiIsImltcG9ydCB7IEFwcCwgUGx1Z2luU2V0dGluZ1RhYiwgU2V0dGluZyB9IGZyb20gJ29ic2lkaWFuJztcbmltcG9ydCBMYXRleEVudmlyb25tZW50cyBmcm9tICcuL21haW4nO1xuXG5leHBvcnQgY2xhc3MgTGF0ZXhFbnZpcm9ubWVudHNTZXR0aW5nVGFiIGV4dGVuZHMgUGx1Z2luU2V0dGluZ1RhYiB7XG4gIHByaXZhdGUgcmVhZG9ubHkgcGx1Z2luOiBMYXRleEVudmlyb25tZW50cztcblxuICBjb25zdHJ1Y3RvcihhcHA6IEFwcCwgcGx1Z2luOiBMYXRleEVudmlyb25tZW50cykge1xuICAgIHN1cGVyKGFwcCwgcGx1Z2luKTtcbiAgICB0aGlzLnBsdWdpbiA9IHBsdWdpbjtcbiAgfVxuXG4gIGRpc3BsYXkoKTogdm9pZCB7XG4gICAgY29uc3QgeyBjb250YWluZXJFbCB9ID0gdGhpcztcblxuICAgIGNvbnRhaW5lckVsLmVtcHR5KCk7XG5cbiAgICBjb250YWluZXJFbC5jcmVhdGVFbCgnaDInLCB7IHRleHQ6ICdTZXR0aW5ncyBmb3IgbGF0ZXggZW52aXJvbm1lbnRzJyB9KTtcblxuICAgIG5ldyBTZXR0aW5nKGNvbnRhaW5lckVsKVxuICAgICAgLnNldE5hbWUoJ0RlZmF1bHQgZW52aXJvbm1lbnQnKVxuICAgICAgLnNldERlc2MoJ1RoZSBkZWZhdWx0IGVudmlyb25tZW50IHRvIGluc2VydCcpXG4gICAgICAuYWRkVGV4dCgodGV4dCkgPT5cbiAgICAgICAgdGV4dFxuICAgICAgICAgIC5zZXRQbGFjZWhvbGRlcignZW52aXJvbm1lbnQnKVxuICAgICAgICAgIC5zZXRWYWx1ZSh0aGlzLnBsdWdpbi5zZXR0aW5ncy5kZWZhdWx0RW52aXJvbm1lbnQpXG4gICAgICAgICAgLm9uQ2hhbmdlKGFzeW5jICh2YWx1ZSkgPT4ge1xuICAgICAgICAgICAgdGhpcy5wbHVnaW4uc2V0dGluZ3MuZGVmYXVsdEVudmlyb25tZW50ID0gdmFsdWU7XG4gICAgICAgICAgICBhd2FpdCB0aGlzLnBsdWdpbi5zYXZlRGF0YSh0aGlzLnBsdWdpbi5zZXR0aW5ncyk7XG4gICAgICAgICAgfSksXG4gICAgICApO1xuXG4gICAgbmV3IFNldHRpbmcoY29udGFpbmVyRWwpXG4gICAgICAuc2V0TmFtZSgnRXh0cmEgZW52aXJvbm1lbnRzJylcbiAgICAgIC5zZXREZXNjKFxuICAgICAgICAnRW52aXJvbm1lbnQgbmFtZXMgdG8gYmUgc3VnZ2VzdGVkIGZvciBjb21wbGV0aW9uIChvbmUgcGVyIGxpbmUpJyxcbiAgICAgIClcbiAgICAgIC5hZGRUZXh0QXJlYSgoYXJlYSkgPT4ge1xuICAgICAgICBhcmVhXG4gICAgICAgICAgLnNldFZhbHVlKHRoaXMucGx1Z2luLnNldHRpbmdzLmN1c3RvbUVudmlyb25tZW50cy5qb2luKCdcXG4nKSlcbiAgICAgICAgICAub25DaGFuZ2UoYXN5bmMgKHZhbHVlKSA9PiB7XG4gICAgICAgICAgICB0aGlzLnBsdWdpbi5zZXR0aW5ncy5jdXN0b21FbnZpcm9ubWVudHMgPSB2YWx1ZVxuICAgICAgICAgICAgICAuc3BsaXQoJ1xcbicpXG4gICAgICAgICAgICAgIC5tYXAoKHgpID0+IHgudHJpbSgpKVxuICAgICAgICAgICAgICAuZmlsdGVyKCh4KSA9PiB4Lmxlbmd0aCA+IDApO1xuICAgICAgICAgICAgYXdhaXQgdGhpcy5wbHVnaW4uc2F2ZURhdGEodGhpcy5wbHVnaW4uc2V0dGluZ3MpO1xuICAgICAgICAgIH0pO1xuICAgICAgfSk7XG4gIH1cbn1cbiIsImltcG9ydCBDb2RlTWlycm9yIGZyb20gJ2NvZGVtaXJyb3InO1xuXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgQWN0aW9uIHtcbiAgY29uc3RydWN0b3IocHVibGljIGRvYzogQ29kZU1pcnJvci5Eb2MpIHt9XG4gIGFic3RyYWN0IHByZXBhcmUoKTogQWN0aW9uO1xuICBhYnN0cmFjdCBleGVjdXRlKGVudk5hbWU6IHN0cmluZyk6IHZvaWQ7XG4gIHB1YmxpYyBzdWdnZXN0TmFtZSgpOiBzdHJpbmcgfCB1bmRlZmluZWQge1xuICAgIHJldHVybiB1bmRlZmluZWQ7XG4gIH1cblxuICBwdWJsaWMgZ2V0IG5lZWRzTmFtZSgpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxufVxuIiwiaW1wb3J0IHsgQWN0aW9uIH0gZnJvbSAnLi9hY3Rpb24nO1xuaW1wb3J0ICogYXMgQ29kZU1pcnJvciBmcm9tICdjb2RlbWlycm9yJztcbmltcG9ydCB7IEVudmlyb25tZW50IH0gZnJvbSAnLi4vZW52aXJvbm1lbnQnO1xuXG5leHBvcnQgY2xhc3MgV3JhcEFjdGlvbiBleHRlbmRzIEFjdGlvbiB7XG4gIGNvbnN0cnVjdG9yKFxuICAgIGRvYzogQ29kZU1pcnJvci5Eb2MsXG4gICAgcHVibGljIHJlYWRvbmx5IGZyb206IENvZGVNaXJyb3IuUG9zaXRpb24sXG4gICAgcHVibGljIHJlYWRvbmx5IHRvOiBDb2RlTWlycm9yLlBvc2l0aW9uLFxuICAgIHB1YmxpYyByZWFkb25seSBhZGRXaGl0ZXNwYWNlID0gdHJ1ZSxcbiAgKSB7XG4gICAgc3VwZXIoZG9jKTtcbiAgfVxuXG4gIHByZXBhcmUoKTogQWN0aW9uIHtcbiAgICByZXR1cm4gdGhpcztcbiAgfVxuXG4gIGV4ZWN1dGUoZW52TmFtZTogc3RyaW5nKTogdm9pZCB7XG4gICAgRW52aXJvbm1lbnQud3JhcChcbiAgICAgIGVudk5hbWUsXG4gICAgICB0aGlzLmRvYyxcbiAgICAgIHRoaXMuZnJvbSxcbiAgICAgIHRoaXMudG8sXG4gICAgICB0aGlzLmFkZFdoaXRlc3BhY2UgPyAnXFxuJyA6ICcnLFxuICAgICk7XG4gIH1cbn1cbiIsImltcG9ydCB7IEFjdGlvbiB9IGZyb20gJy4vYWN0aW9uJztcbmltcG9ydCB7IFdyYXBBY3Rpb24gfSBmcm9tICcuL3dyYXBBY3Rpb24nO1xuaW1wb3J0IHsgRW52aXJvbm1lbnQgfSBmcm9tICcuLi9lbnZpcm9ubWVudCc7XG5cbmV4cG9ydCBjbGFzcyBJbnNlcnRBY3Rpb24gZXh0ZW5kcyBBY3Rpb24ge1xuICBwcmVwYXJlKCk6IEFjdGlvbiB7XG4gICAgaWYgKHRoaXMuZG9jLnNvbWV0aGluZ1NlbGVjdGVkKCkpIHtcbiAgICAgIHJldHVybiBuZXcgV3JhcEFjdGlvbihcbiAgICAgICAgdGhpcy5kb2MsXG4gICAgICAgIHRoaXMuZG9jLmdldEN1cnNvcignZnJvbScpLFxuICAgICAgICB0aGlzLmRvYy5nZXRDdXJzb3IoJ3RvJyksXG4gICAgICApLnByZXBhcmUoKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXM7XG4gIH1cblxuICBleGVjdXRlKGVudk5hbWU6IHN0cmluZyk6IHZvaWQge1xuICAgIEVudmlyb25tZW50LmNyZWF0ZShlbnZOYW1lLCB0aGlzLmRvYywgdGhpcy5kb2MuZ2V0Q3Vyc29yKCkpO1xuICB9XG59XG4iLCJpbXBvcnQgeyBBY3Rpb24gfSBmcm9tICcuL2FjdGlvbic7XG5pbXBvcnQgeyBNYXRoQmxvY2sgfSBmcm9tICcuLi9tYXRoYmxvY2snO1xuaW1wb3J0IHsgRW52aXJvbm1lbnQgfSBmcm9tICcuLi9lbnZpcm9ubWVudCc7XG5pbXBvcnQgeyBXcmFwQWN0aW9uIH0gZnJvbSAnLi93cmFwQWN0aW9uJztcblxuZXhwb3J0IGNsYXNzIENoYW5nZUFjdGlvbiBleHRlbmRzIEFjdGlvbiB7XG4gIHByaXZhdGUgY3VycmVudDogRW52aXJvbm1lbnQgfCB1bmRlZmluZWQ7XG4gIHByaXZhdGUgbmFtZTogc3RyaW5nIHwgdW5kZWZpbmVkO1xuXG4gIHB1YmxpYyBzdWdnZXN0TmFtZSgpOiBzdHJpbmcgfCB1bmRlZmluZWQge1xuICAgIHJldHVybiB0aGlzLm5hbWU7XG4gIH1cblxuICBwcmVwYXJlKCk6IEFjdGlvbiB7XG4gICAgY29uc3QgY3Vyc29yID0gdGhpcy5kb2MuZ2V0Q3Vyc29yKCk7XG4gICAgY29uc3QgYmxvY2sgPSBuZXcgTWF0aEJsb2NrKHRoaXMuZG9jLCBjdXJzb3IpO1xuICAgIHRoaXMuY3VycmVudCA9IGJsb2NrLmdldEVuY2xvc2luZ0Vudmlyb25tZW50KGN1cnNvcik7XG4gICAgaWYgKHRoaXMuY3VycmVudCA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICByZXR1cm4gbmV3IFdyYXBBY3Rpb24oXG4gICAgICAgIHRoaXMuZG9jLFxuICAgICAgICBibG9jay5zdGFydFBvc2l0aW9uLFxuICAgICAgICBibG9jay5lbmRQb3NpdGlvbixcbiAgICAgICAgYmxvY2suc3RhcnRQb3NpdGlvbi5saW5lID09PSBibG9jay5lbmRQb3NpdGlvbi5saW5lLFxuICAgICAgKTtcbiAgICB9XG4gICAgdGhpcy5uYW1lID0gdGhpcy5jdXJyZW50Lm5hbWU7XG4gICAgcmV0dXJuIHRoaXM7XG4gIH1cblxuICBleGVjdXRlKGVudk5hbWU6IHN0cmluZyk6IHZvaWQge1xuICAgIGlmICh0aGlzLmN1cnJlbnQgIT09IHVuZGVmaW5lZCkgdGhpcy5jdXJyZW50LnJlcGxhY2UoZW52TmFtZSk7XG4gIH1cbn1cbiIsImltcG9ydCB7IEFjdGlvbiB9IGZyb20gJy4vYWN0aW9uJztcbmltcG9ydCB7IE1hdGhCbG9jayB9IGZyb20gJy4uL21hdGhibG9jayc7XG5pbXBvcnQgeyBFbnZpcm9ubWVudCB9IGZyb20gJy4uL2Vudmlyb25tZW50JztcblxuZXhwb3J0IGNsYXNzIERlbGV0ZUFjdGlvbiBleHRlbmRzIEFjdGlvbiB7XG4gIHByaXZhdGUgY3VycmVudDogRW52aXJvbm1lbnQgfCB1bmRlZmluZWQ7XG5cbiAgcHVibGljIGdldCBuZWVkc05hbWUoKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG5cbiAgcHJlcGFyZSgpOiBBY3Rpb24ge1xuICAgIGNvbnN0IGN1cnNvciA9IHRoaXMuZG9jLmdldEN1cnNvcigpO1xuICAgIGNvbnN0IGJsb2NrID0gbmV3IE1hdGhCbG9jayh0aGlzLmRvYywgY3Vyc29yKTtcbiAgICB0aGlzLmN1cnJlbnQgPSBibG9jay5nZXRFbmNsb3NpbmdFbnZpcm9ubWVudChjdXJzb3IpO1xuICAgIHJldHVybiB0aGlzO1xuICB9XG5cbiAgZXhlY3V0ZShfZW52TmFtZTogc3RyaW5nKTogdm9pZCB7XG4gICAgaWYgKHRoaXMuY3VycmVudCAhPT0gdW5kZWZpbmVkKSB0aGlzLmN1cnJlbnQudW53cmFwKCk7XG4gIH1cbn1cbiIsImltcG9ydCB7IE1hcmtkb3duVmlldywgTm90aWNlLCBQbHVnaW4gfSBmcm9tICdvYnNpZGlhbic7XG5pbXBvcnQgeyBlbnN1cmVTZXR0aW5ncywgTGF0ZXhFbnZpcm9ubWVudHNTZXR0aW5ncyB9IGZyb20gJy4vc2V0dGluZ3MnO1xuaW1wb3J0IENvZGVNaXJyb3IgZnJvbSAnY29kZW1pcnJvcic7XG5pbXBvcnQgeyBNYXRoQmxvY2sgfSBmcm9tICcuL21hdGhibG9jayc7XG5pbXBvcnQgeyBFbnZNb2RhbCB9IGZyb20gJy4vZW52bW9kYWwnO1xuaW1wb3J0IHsgTGF0ZXhFbnZpcm9ubWVudHNTZXR0aW5nVGFiIH0gZnJvbSAnLi9sYXRleEVudmlyb25tZW50c1NldHRpbmdzVGFiJztcbmltcG9ydCB7IEluc2VydEFjdGlvbiB9IGZyb20gJy4vYWN0aW9ucy9pbnNlcnRBY3Rpb24nO1xuaW1wb3J0IHsgQ2hhbmdlQWN0aW9uIH0gZnJvbSAnLi9hY3Rpb25zL2NoYW5nZUFjdGlvbic7XG5pbXBvcnQgeyBBY3Rpb24gfSBmcm9tICcuL2FjdGlvbnMvYWN0aW9uJztcbmltcG9ydCB7IERlbGV0ZUFjdGlvbiB9IGZyb20gJy4vYWN0aW9ucy9kZWxldGVBY3Rpb24nO1xuXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBMYXRleEVudmlyb25tZW50cyBleHRlbmRzIFBsdWdpbiB7XG4gIHB1YmxpYyBzZXR0aW5nczogTGF0ZXhFbnZpcm9ubWVudHNTZXR0aW5ncyA9IG5ldyBMYXRleEVudmlyb25tZW50c1NldHRpbmdzKCk7XG5cbiAgYXN5bmMgb25sb2FkKCk6IFByb21pc2U8dm9pZD4ge1xuICAgIGNvbnN0IHNldHRpbmdzID0gYXdhaXQgdGhpcy5sb2FkRGF0YSgpO1xuICAgIGlmIChzZXR0aW5ncyAhPT0gbnVsbCkge1xuICAgICAgdGhpcy5zZXR0aW5ncyA9IGVuc3VyZVNldHRpbmdzKHNldHRpbmdzKTtcbiAgICB9XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6ICdpbnNlcnQtbGF0ZXgtZW52JyxcbiAgICAgIG5hbWU6ICdJbnNlcnQgTGFUZVggZW52aXJvbm1lbnQnLFxuICAgICAgY2hlY2tDYWxsYmFjazogdGhpcy5tYXRoTW9kZUNhbGxiYWNrKEluc2VydEFjdGlvbiksXG4gICAgfSk7XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6ICdjaGFuZ2UtbGF0ZXgtZW52JyxcbiAgICAgIG5hbWU6ICdDaGFuZ2UgTGFUZVggZW52aXJvbm1lbnQnLFxuICAgICAgY2hlY2tDYWxsYmFjazogdGhpcy5tYXRoTW9kZUNhbGxiYWNrKENoYW5nZUFjdGlvbiksXG4gICAgfSk7XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6ICdkZWxldGUtbGF0ZXgtZW52JyxcbiAgICAgIG5hbWU6ICdEZWxldGUgTGFUZVggZW52aXJvbm1lbnQnLFxuICAgICAgY2hlY2tDYWxsYmFjazogdGhpcy5tYXRoTW9kZUNhbGxiYWNrKERlbGV0ZUFjdGlvbiksXG4gICAgfSk7XG5cbiAgICB0aGlzLmFkZFNldHRpbmdUYWIobmV3IExhdGV4RW52aXJvbm1lbnRzU2V0dGluZ1RhYih0aGlzLmFwcCwgdGhpcykpO1xuICB9XG5cbiAgcHJpdmF0ZSBtYXRoTW9kZUNhbGxiYWNrPEEgZXh0ZW5kcyBBY3Rpb24+KFxuICAgIEFjdGlvblR5cGU6IG5ldyAoZG9jOiBDb2RlTWlycm9yLkRvYykgPT4gQSxcbiAgKSB7XG4gICAgcmV0dXJuIChjaGVja2luZzogYm9vbGVhbikgPT4ge1xuICAgICAgY29uc3QgbGVhZiA9IHRoaXMuYXBwLndvcmtzcGFjZS5hY3RpdmVMZWFmO1xuICAgICAgaWYgKGxlYWYudmlldyBpbnN0YW5jZW9mIE1hcmtkb3duVmlldykge1xuICAgICAgICBjb25zdCBlZGl0b3IgPSBsZWFmLnZpZXcuc291cmNlTW9kZS5jbUVkaXRvcjtcbiAgICAgICAgY29uc3QgY3Vyc29yID0gZWRpdG9yLmdldEN1cnNvcigpO1xuXG4gICAgICAgIGlmICghTWF0aEJsb2NrLmlzTWF0aE1vZGUoY3Vyc29yLCBlZGl0b3IpKSB7XG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYgKCFjaGVja2luZykge1xuICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICBjb25zdCBhY3Rpb24gPSBuZXcgQWN0aW9uVHlwZShlZGl0b3IuZ2V0RG9jKCkpLnByZXBhcmUoKTtcbiAgICAgICAgICAgIHRoaXMud2l0aFByb21wdE5hbWUoZWRpdG9yLCBhY3Rpb24pO1xuICAgICAgICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgICAgICAgIC8qIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSBuby1uZXcgKi9cbiAgICAgICAgICAgIG5ldyBOb3RpY2UoZS5tZXNzYWdlKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICB9XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfTtcbiAgfVxuXG4gIHByaXZhdGUgd2l0aFByb21wdE5hbWUoZWRpdG9yOiBDb2RlTWlycm9yLkVkaXRvciwgYWN0aW9uOiBBY3Rpb24pOiB2b2lkIHtcbiAgICBjb25zdCBjYWxsID0gKGVudk5hbWU6IHN0cmluZyk6IHZvaWQgPT4ge1xuICAgICAgZWRpdG9yLm9wZXJhdGlvbigoKSA9PiBhY3Rpb24uZXhlY3V0ZShlbnZOYW1lKSk7XG4gICAgICBlZGl0b3IuZm9jdXMoKTtcbiAgICB9O1xuXG4gICAgaWYgKGFjdGlvbi5uZWVkc05hbWUpIHtcbiAgICAgIGNvbnN0IHN1Z2dlc3RlZCA9IGFjdGlvbi5zdWdnZXN0TmFtZSgpO1xuICAgICAgRW52TW9kYWwuY2FsbGJhY2soXG4gICAgICAgIHRoaXMuYXBwLFxuICAgICAgICB0aGlzLnNldHRpbmdzLFxuICAgICAgICBzdWdnZXN0ZWQgIT09IHVuZGVmaW5lZCA/IHN1Z2dlc3RlZCA6IHRoaXMuc2V0dGluZ3MuZGVmYXVsdEVudmlyb25tZW50LFxuICAgICAgICBjYWxsLFxuICAgICAgKTtcbiAgICB9IGVsc2Uge1xuICAgICAgY2FsbCgnKicpO1xuICAgIH1cbiAgfVxufVxuIl0sIm5hbWVzIjpbIkZ1enp5U3VnZ2VzdE1vZGFsIiwiU2V0dGluZyIsIlBsdWdpblNldHRpbmdUYWIiLCJNYXJrZG93blZpZXciLCJOb3RpY2UiLCJQbHVnaW4iXSwibWFwcGluZ3MiOiI7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLElBQUksYUFBYSxHQUFHLFNBQVMsQ0FBQyxFQUFFLENBQUMsRUFBRTtBQUNuQyxJQUFJLGFBQWEsR0FBRyxNQUFNLENBQUMsY0FBYztBQUN6QyxTQUFTLEVBQUUsU0FBUyxFQUFFLEVBQUUsRUFBRSxZQUFZLEtBQUssSUFBSSxVQUFVLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsU0FBUyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUM7QUFDcEYsUUFBUSxVQUFVLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxLQUFLLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxJQUFJLE1BQU0sQ0FBQyxTQUFTLENBQUMsY0FBYyxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUM7QUFDMUcsSUFBSSxPQUFPLGFBQWEsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7QUFDL0IsQ0FBQyxDQUFDO0FBQ0Y7QUFDTyxTQUFTLFNBQVMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFO0FBQ2hDLElBQUksYUFBYSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztBQUN4QixJQUFJLFNBQVMsRUFBRSxHQUFHLEVBQUUsSUFBSSxDQUFDLFdBQVcsR0FBRyxDQUFDLENBQUMsRUFBRTtBQUMzQyxJQUFJLENBQUMsQ0FBQyxTQUFTLEdBQUcsQ0FBQyxLQUFLLElBQUksR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxTQUFTLEdBQUcsQ0FBQyxDQUFDLFNBQVMsRUFBRSxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUM7QUFDekYsQ0FBQztBQXVDRDtBQUNPLFNBQVMsU0FBUyxDQUFDLE9BQU8sRUFBRSxVQUFVLEVBQUUsQ0FBQyxFQUFFLFNBQVMsRUFBRTtBQUM3RCxJQUFJLFNBQVMsS0FBSyxDQUFDLEtBQUssRUFBRSxFQUFFLE9BQU8sS0FBSyxZQUFZLENBQUMsR0FBRyxLQUFLLEdBQUcsSUFBSSxDQUFDLENBQUMsVUFBVSxPQUFPLEVBQUUsRUFBRSxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRTtBQUNoSCxJQUFJLE9BQU8sS0FBSyxDQUFDLEtBQUssQ0FBQyxHQUFHLE9BQU8sQ0FBQyxFQUFFLFVBQVUsT0FBTyxFQUFFLE1BQU0sRUFBRTtBQUMvRCxRQUFRLFNBQVMsU0FBUyxDQUFDLEtBQUssRUFBRSxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsU0FBUyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUU7QUFDbkcsUUFBUSxTQUFTLFFBQVEsQ0FBQyxLQUFLLEVBQUUsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLFNBQVMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUU7QUFDdEcsUUFBUSxTQUFTLElBQUksQ0FBQyxNQUFNLEVBQUUsRUFBRSxNQUFNLENBQUMsSUFBSSxHQUFHLE9BQU8sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLEdBQUcsS0FBSyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxFQUFFLFFBQVEsQ0FBQyxDQUFDLEVBQUU7QUFDdEgsUUFBUSxJQUFJLENBQUMsQ0FBQyxTQUFTLEdBQUcsU0FBUyxDQUFDLEtBQUssQ0FBQyxPQUFPLEVBQUUsVUFBVSxJQUFJLEVBQUUsQ0FBQyxFQUFFLElBQUksRUFBRSxDQUFDLENBQUM7QUFDOUUsS0FBSyxDQUFDLENBQUM7QUFDUCxDQUFDO0FBQ0Q7QUFDTyxTQUFTLFdBQVcsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFO0FBQzNDLElBQUksSUFBSSxDQUFDLEdBQUcsRUFBRSxLQUFLLEVBQUUsQ0FBQyxFQUFFLElBQUksRUFBRSxXQUFXLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLElBQUksRUFBRSxFQUFFLEVBQUUsR0FBRyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQztBQUNySCxJQUFJLE9BQU8sQ0FBQyxHQUFHLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxPQUFPLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLFFBQVEsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxPQUFPLE1BQU0sS0FBSyxVQUFVLEtBQUssQ0FBQyxDQUFDLE1BQU0sQ0FBQyxRQUFRLENBQUMsR0FBRyxXQUFXLEVBQUUsT0FBTyxJQUFJLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDO0FBQzdKLElBQUksU0FBUyxJQUFJLENBQUMsQ0FBQyxFQUFFLEVBQUUsT0FBTyxVQUFVLENBQUMsRUFBRSxFQUFFLE9BQU8sSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUU7QUFDdEUsSUFBSSxTQUFTLElBQUksQ0FBQyxFQUFFLEVBQUU7QUFDdEIsUUFBUSxJQUFJLENBQUMsRUFBRSxNQUFNLElBQUksU0FBUyxDQUFDLGlDQUFpQyxDQUFDLENBQUM7QUFDdEUsUUFBUSxPQUFPLENBQUMsRUFBRSxJQUFJO0FBQ3RCLFlBQVksSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsUUFBUSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsUUFBUSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMs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
