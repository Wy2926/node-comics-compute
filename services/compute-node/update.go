package main

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const releaseAPI = "https://api.github.com/repos/Wy2926/node-comics-compute/releases/latest"
const releasePrefix = "https://github.com/Wy2926/node-comics-compute/releases/download/"

var releaseVersion = regexp.MustCompile(`^v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[A-Za-z0-9.-]+)?$`)

func newerRelease(next, current string) (bool, error) {
	n, c := releaseVersion.FindStringSubmatch(next), releaseVersion.FindStringSubmatch(current)
	if n == nil || c == nil || n[4] != "" {
		return false, errors.New("release version must be a stable major.minor.patch version")
	}
	for i := 1; i <= 3; i++ {
		nv, ne := strconv.ParseUint(n[i], 10, 32)
		cv, ce := strconv.ParseUint(c[i], 10, 32)
		if ne != nil || ce != nil {
			return false, errors.New("invalid release version number")
		}
		if nv != cv {
			return nv > cv, nil
		}
	}
	return c[4] != "", nil
}

// All remote input is bounded and HTTPS-only, including redirects. Only public
// release URLs are used; node identity never enters these requests or logs.
func fetchUpdate(ctx context.Context, address string, output io.Writer, limit int64, label string) error {
	u, err := url.Parse(address)
	if err != nil || u.Scheme != "https" || u.User != nil {
		return errors.New("update requires an HTTPS release URL")
	}
	client := &http.Client{Timeout: 20 * time.Minute, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 10 || req.URL.Scheme != "https" || req.URL.User != nil {
			return errors.New("invalid release redirect")
		}
		return nil
	}}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "NodeComics-Updater/1")
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("cannot download %s; check the network and retry Update.cmd", label)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: HTTP %d; a published release with ZIP and SHA-256 assets is required", label, resp.StatusCode)
	}
	if resp.ContentLength > limit {
		return errors.New("release download exceeds size limit")
	}
	reader := io.LimitReader(resp.Body, limit+1)
	buffer := make([]byte, 1024*1024)
	var received int64
	reported := time.Time{}
	for {
		n, readErr := reader.Read(buffer)
		received += int64(n)
		if received > limit {
			return errors.New("release download exceeds size limit")
		}
		if _, err := output.Write(buffer[:n]); err != nil {
			return err
		}
		if limit > 4*1024*1024 && (time.Since(reported) >= time.Second || readErr == io.EOF) {
			fmt.Printf("  %s: %.1f MiB", label, float64(received)/(1024*1024))
			if resp.ContentLength > 0 {
				fmt.Printf(" / %.1f MiB (%d%%)", float64(resp.ContentLength)/(1024*1024), received*100/resp.ContentLength)
			}
			fmt.Println()
			reported = time.Now()
		}
		if readErr == io.EOF {
			return nil
		}
		if readErr != nil {
			return fmt.Errorf("%s download interrupted; retry Update.cmd", label)
		}
	}
}

func downloadUpdate(ctx context.Context, work, current string) (string, string, error) {
	var metadata strings.Builder
	if err := fetchUpdate(ctx, releaseAPI, &metadata, 1024*1024, "latest release"); err != nil {
		return "", "", err
	}
	var release struct {
		Tag        string `json:"tag_name"`
		Draft      bool   `json:"draft"`
		Prerelease bool   `json:"prerelease"`
		Assets     []struct {
			Name string `json:"name"`
			URL  string `json:"browser_download_url"`
		} `json:"assets"`
	}
	if err := json.Unmarshal([]byte(metadata.String()), &release); err != nil {
		return "", "", err
	}
	newer, err := newerRelease(release.Tag, current)
	if err != nil || release.Draft || release.Prerelease {
		return "", "", errors.New("latest release metadata has no valid stable version")
	}
	if !newer {
		fmt.Printf("Already up to date: %s (latest: %s).\n", current, release.Tag)
		return "", "", nil
	}
	fmt.Printf("  %s -> %s\n", current, release.Tag)
	assets := map[string]string{}
	for _, asset := range release.Assets {
		if strings.HasPrefix(asset.URL, releasePrefix) {
			assets[asset.Name] = asset.URL
		}
	}
	if assets["NodeComicsNode.zip"] == "" || assets["NodeComicsNode.zip.sha256"] == "" {
		return "", "", errors.New("release must publish NodeComicsNode.zip and NodeComicsNode.zip.sha256")
	}
	var checksum strings.Builder
	if err := fetchUpdate(ctx, assets["NodeComicsNode.zip.sha256"], &checksum, 1024, "release checksum"); err != nil {
		return "", "", err
	}
	fields := strings.Fields(checksum.String())
	if len(fields) != 2 || fields[1] != "NodeComicsNode.zip" || len(fields[0]) != 64 {
		return "", "", errors.New("invalid release checksum file")
	}
	archive := filepath.Join(work, "release.zip")
	file, err := os.Create(archive)
	if err != nil {
		return "", "", err
	}
	hash := sha256.New()
	err = fetchUpdate(ctx, assets["NodeComicsNode.zip"], io.MultiWriter(file, hash), 256*1024*1024, "release ZIP")
	err = errors.Join(err, file.Close())
	if err != nil {
		return "", "", err
	}
	if hex.EncodeToString(hash.Sum(nil)) != fields[0] {
		return "", "", errors.New("release ZIP checksum mismatch; installed version is unchanged")
	}
	return archive, strings.TrimPrefix(release.Tag, "v"), nil
}

func releasePath(name string) bool {
	parts := strings.Split(name, "/")
	for _, part := range parts {
		if part == "" || part == "." || part == ".." || strings.TrimRight(part, ". ") != part || strings.ContainsAny(part, `\:<>"|?*`) {
			return false
		}
		base := strings.ToUpper(strings.SplitN(part, ".", 2)[0])
		if base == "CON" || base == "PRN" || base == "AUX" || base == "NUL" ||
			(len(base) == 4 && (strings.HasPrefix(base, "COM") || strings.HasPrefix(base, "LPT")) && base[3] >= '0' && base[3] <= '9') {
			return false
		}
	}
	switch parts[0] {
	case "source", "packages", "docs", "licenses":
		return true
	case "node.exe", "portable.json", "README.md", "LICENSE", "Start.cmd", "Stop.cmd", "Status.cmd", "Update.cmd", "Enable-Autostart.cmd", "Disable-Autostart.cmd":
		return len(parts) == 1
	}
	return false
}

func unpackUpdate(archive, destination, version string) (releaseManifest, error) {
	var manifest releaseManifest
	packageFile, err := zip.OpenReader(archive)
	if err != nil {
		return manifest, err
	}
	defer packageFile.Close()
	if len(packageFile.File) > 20000 {
		return manifest, errors.New("too many release entries")
	}
	seen := map[string]bool{}
	var total uint64
	for _, entry := range packageFile.File {
		name := strings.TrimPrefix(entry.Name, "NodeComicsNode/")
		if entry.Mode().IsDir() {
			name = strings.TrimSuffix(name, "/")
		}
		if !strings.HasPrefix(entry.Name, "NodeComicsNode/") || !releasePath(name) || seen[strings.ToLower(name)] {
			return manifest, errors.New("unsafe or duplicate release entry")
		}
		seen[strings.ToLower(name)] = true
		if entry.Mode().IsDir() {
			continue
		}
		total += entry.UncompressedSize64
		if !entry.Mode().IsRegular() || entry.UncompressedSize64 > 128*1024*1024 || total > 512*1024*1024 || (name == "portable.json" && entry.UncompressedSize64 > 4*1024*1024) {
			return manifest, errors.New("invalid release entry type or size")
		}
		path := filepath.Join(destination, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			return manifest, err
		}
		source, err := entry.Open()
		if err != nil {
			return manifest, err
		}
		target, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			source.Close()
			return manifest, err
		}
		_, err = io.Copy(target, io.LimitReader(source, int64(entry.UncompressedSize64)+1))
		err = errors.Join(err, target.Close(), source.Close())
		if err != nil {
			return manifest, err
		}
	}
	if err := readJSON(filepath.Join(destination, "portable.json"), &manifest); err != nil {
		return manifest, err
	}
	if manifest.Version != version || manifest.Format != "gpu-portable-v1" || manifest.Platform != "windows-amd64" || manifest.Python == "" || len(manifest.Requirements) != 2 {
		return manifest, errors.New("release manifest does not match the published Windows version")
	}
	for _, backend := range []string{"cuda", "amd"} {
		digest, err := hex.DecodeString(manifest.Requirements[backend])
		if err != nil || len(digest) != sha256.Size {
			return manifest, errors.New("invalid release dependency digest")
		}
	}
	for _, name := range []string{"node.exe", "Start.cmd", "Update.cmd", "source/compute-node/update.ps1", "source/compute-node/prepare.ps1", "source/compute-node/run_node.py", "source/classic-engine/classic_node/__main__.py"} {
		if manifest.Files[name] == "" {
			return manifest, fmt.Errorf("release is missing %s", name)
		}
	}
	for name := range manifest.Files {
		if !releasePath(name) || name == "portable.json" {
			return manifest, errors.New("unsafe manifest path")
		}
	}
	for _, entry := range packageFile.File {
		name := strings.TrimPrefix(entry.Name, "NodeComicsNode/")
		if !entry.Mode().IsDir() && name != "portable.json" && manifest.Files[name] == "" {
			return manifest, errors.New("release contains an untracked file")
		}
	}
	return manifest, verifyFiles(destination, manifest.Files)
}

// Top-level directory renames keep code snapshots coherent. The journal is
// published before the first move; Update.cmd rolls back interrupted switches.
type updateTransaction struct {
	Existing map[string]bool `json:"existing"`
}

func (a app) updateWork() string { return filepath.Join(a.root, ".setup", "update") }
func (a app) updatePending() bool {
	_, err := os.Stat(filepath.Join(a.updateWork(), "transaction.json"))
	return !os.IsNotExist(err)
}

func (a app) restoreUpdate() error {
	var transaction updateTransaction
	journal := filepath.Join(a.updateWork(), "transaction.json")
	if err := readJSON(journal, &transaction); err != nil {
		return err
	}
	if len(transaction.Existing) == 0 {
		return errors.New("empty update recovery journal")
	}
	for name := range transaction.Existing {
		if strings.ContainsAny(name, `/\`) || (!releasePath(name) && name != "payload") {
			return errors.New("invalid update recovery path")
		}
	}
	for name, existed := range transaction.Existing {
		backup := filepath.Join(a.updateWork(), "previous", name)
		target := filepath.Join(a.root, name)
		info, err := os.Lstat(backup)
		if err != nil && !os.IsNotExist(err) {
			return err
		}
		if (err == nil && info.IsDir()) || !existed {
			if err := os.RemoveAll(target); err != nil {
				return err
			}
		}
		if err == nil {
			if err := os.Rename(backup, target); err != nil {
				return err
			}
		}
	}
	return os.Remove(journal)
}

func backupUpdateFile(source, target string) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	temporary := target + ".copying"
	output, err := os.Create(temporary)
	if err != nil {
		return err
	}
	_, err = io.Copy(output, input)
	err = errors.Join(err, output.Sync(), output.Close())
	if err != nil {
		return err
	}
	return os.Rename(temporary, target)
}

func (a app) switchUpdate(stage string, old, next releaseManifest, replacePayload bool) error {
	items := map[string]bool{"portable.json": true}
	oldItems := map[string]bool{"portable.json": true, "payload": true}
	for name := range old.Files {
		oldItems[strings.SplitN(name, "/", 2)[0]] = true
	}
	for _, manifest := range []releaseManifest{old, next} {
		for name := range manifest.Files {
			if !releasePath(name) {
				return errors.New("invalid installed release path")
			}
			items[strings.SplitN(name, "/", 2)[0]] = true
		}
	}
	if replacePayload {
		items["payload"] = true
	}
	transaction := updateTransaction{Existing: map[string]bool{}}
	names := []string{}
	for name := range items {
		path := filepath.Join(a.root, name)
		_, err := os.Lstat(path)
		if err != nil && !os.IsNotExist(err) {
			return err
		}
		transaction.Existing[name] = err == nil
		if err == nil {
			if !oldItems[name] {
				return fmt.Errorf("new release would overwrite an unmanaged file: %s", name)
			}
			real, err := filepath.EvalSymlinks(path)
			if err != nil || !strings.EqualFold(real, path) {
				return errors.New("update target must not be a link")
			}
		}
		names = append(names, name)
	}
	sort.Strings(names)
	previous := filepath.Join(a.updateWork(), "previous")
	if err := os.MkdirAll(previous, 0700); err != nil {
		return err
	}
	if entries, err := os.ReadDir(previous); err != nil || len(entries) != 0 {
		return errors.New("previous update backup must be empty before switching")
	}
	journal := filepath.Join(a.updateWork(), "transaction.json")
	if err := writeJSON(journal, transaction); err != nil {
		return err
	}
	apply := func() error {
		for _, name := range names {
			target := filepath.Join(a.root, name)
			if transaction.Existing[name] {
				info, err := os.Stat(target)
				if err != nil {
					return err
				}
				backup := filepath.Join(previous, name)
				if info.IsDir() {
					err = os.Rename(target, backup)
				} else {
					err = backupUpdateFile(target, backup)
				}
				if err != nil {
					return err
				}
			}
			source := filepath.Join(stage, name)
			if _, err := os.Stat(source); os.IsNotExist(err) {
				if err := os.RemoveAll(target); err != nil {
					return err
				}
				continue // Removed by the new release.
			}
			if err := os.Rename(source, target); err != nil {
				return err
			}
		}
		if err := a.verifySeed(); err != nil {
			return err
		}
		if _, err := os.Stat(a.bundleRoot()); err == nil {
			if err := a.checkPayload(""); err != nil {
				return err
			}
		}
		return os.Remove(journal) // Commit; backups remain until the next update.
	}
	if err := apply(); err != nil {
		return errors.Join(err, a.restoreUpdate())
	}
	return nil
}

func (a app) ensureUpdateDrained() error {
	if _, err := os.Stat(a.configPath()); os.IsNotExist(err) {
		return nil
	}
	cmd := a.worker("pending")
	if err := runContained(cmd, a.jobName("prepare")); err != nil {
		return errors.New("unfinished tasks or deliveries remain; the current version is preserved. Let it finish/recover, then click Update again")
	}
	var config struct {
		StateDir string `json:"state_dir"`
	}
	if err := readJSON(a.configPath(), &config); err != nil {
		return err
	}
	state := config.StateDir
	if state == "" {
		state = "state"
	}
	if !filepath.IsAbs(state) {
		state = filepath.Join(a.home, state)
	}
	entries, err := os.ReadDir(filepath.Join(state, "documents"))
	if (err != nil && !os.IsNotExist(err)) || len(entries) > 0 {
		return errors.New("document workspaces still need recovery; finish them with the current version before updating")
	}
	return nil
}

func (a app) update() error {
	finish, err := a.beginLaunch("updating")
	if err != nil {
		return err
	}
	defer finish()
	if a.updatePending() {
		fmt.Println("Recovering an interrupted update...")
		if err := a.stopHost(); err != nil {
			return err
		}
		lock, err := acquireLock(filepath.Join(a.root, ".setup", "runtime.lock"))
		if err != nil {
			return errors.New("another data directory still uses this release; stop its node before recovery")
		}
		defer lock.Close()
		if err := a.restoreUpdate(); err != nil {
			return err
		}
		fmt.Println("Previous version restored. Click Start.cmd to run it, or Update.cmd to try again.")
		return nil
	}
	var old releaseManifest
	if err := readJSON(filepath.Join(a.root, "portable.json"), &old); err != nil {
		return err
	}
	work := a.updateWork()
	if err := os.RemoveAll(work); err != nil {
		return err
	}
	if err := os.MkdirAll(work, 0700); err != nil {
		return err
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	fmt.Println("[1/5] Checking the latest stable release...")
	archive, nextVersion, err := downloadUpdate(ctx, work, old.Version)
	if err != nil || archive == "" {
		return err
	}
	fmt.Println("[2/5] Verifying and extracting the new version...")
	stage := filepath.Join(work, "next")
	next, err := unpackUpdate(archive, stage, nextVersion)
	if err != nil {
		return err
	}
	fmt.Println("[3/5] Checking the installed GPU environment...")
	replacePayload := false
	if _, err := os.Stat(a.bundleRoot()); err == nil {
		installed, err := a.manifest()
		if err != nil {
			return err
		}
		replacePayload = installed.Python != next.Python || old.Requirements[installed.Backend] != next.Requirements[installed.Backend]
		for _, name := range []string{"source/classic-engine/mtu_engine/upstream.lock.json", "source/compute-node/assets.json"} {
			replacePayload = replacePayload || old.Files[name] != next.Files[name]
		}
		if replacePayload {
			fmt.Println("Dependencies/assets changed; preparing the new environment from the verified cache...")
			if err := a.prepareEnvironment(stage, installed.Backend); err != nil {
				return err
			}
		} else {
			if err := a.checkPayload(""); err != nil {
				return err
			}
			fmt.Println("  Reusing the installed runtime, models and fonts.")
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	fmt.Println("[4/5] Stopping the worker and checking pending deliveries...")
	_, wasRunning := a.liveHost()
	if err := a.stopHost(); err != nil {
		return err
	}
	// Also excludes hosts using a different --home in the same package.
	runtime, err := acquireLock(filepath.Join(a.root, ".setup", "runtime.lock"))
	if err != nil {
		return errors.New("another data directory still uses this release; stop its node before updating")
	}
	err = a.ensureUpdateDrained()
	if err == nil {
		err = ctx.Err()
	}
	if err == nil {
		err = a.switchUpdate(stage, old, next, replacePayload)
	}
	runtime.Close()
	if err != nil {
		if wasRunning && !a.updatePending() {
			err = errors.Join(err, a.startBackground(false))
		}
		return err
	}
	fmt.Printf("[5/5] Updated to %s. Node identity and recovery data are preserved.\n", nextVersion)
	if wasRunning {
		fmt.Println("Restarting the node and waiting for a fresh heartbeat...")
		return a.startBackground(true)
	}
	fmt.Println("Double-click Start.cmd when ready to start the node.")
	return nil
}

func (a app) prepareEnvironment(root, backend string) error {
	setup := filepath.Join(a.root, ".setup")
	powershell := filepath.Join(os.Getenv("SystemRoot"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
	cmd := exec.Command(powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", filepath.Join(root, "source/compute-node/prepare.ps1"), "-Root", root, "-Backend", backend, "-WorkDirectory", setup)
	cmd.Dir, cmd.Env = root, localEnvironment(setup)
	if err := localDirectories(setup); err != nil {
		return err
	}
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := runContained(cmd, a.jobName("prepare")); err != nil {
		return fmt.Errorf("preparation failed; verified downloads retained: %w", err)
	}
	return nil
}
